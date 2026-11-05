//! GNU vector (`vector_size`) expression lowering.
//!
//! Two storage stories, per [`crate::ty::lower_ty`]: a 16-byte vector
//! is a first-class MIR vector, so element-wise arithmetic, splats, and
//! lane masks emit the same shape-generic instructions the vectorizer
//! produces. Every smaller shape (`__m64`, `__m32`, `__v2sf`, ...)
//! bit-puns to an integer of the vector's byte size — the MMX class's
//! own register story — so movement, storage, and the ABI ride the
//! image integer and only the lane-wise operators expand here: bitwise
//! operators are lane-exact on the image as it stands, while arithmetic
//! (which carries or borrows across lane boundaries), division,
//! remainder, shifts, and comparisons extract the lanes, compute at
//! lane width, and rejoin. Float lanes bitcast through the image's
//! integer lanes.

use super::*;
use glcc_frontend::check::CheckedExpr;

/// The vector facts lowering needs: the element's frontend and MIR
/// types, the lane count, and the total byte size.
pub(crate) struct VecShape {
    pub(crate) elem_fe: fe::parse::Ty,
    pub(crate) elem_mir: mir::Ty,
    pub(crate) lanes: u32,
    pub(crate) bytes: u64,
}

/// Bit width of a bit-punned vector's integer image (2-, 4-, or 8-byte
/// shapes; 16-byte vectors never take this path).
fn image_bits(shape: &VecShape) -> u32 {
    (shape.bytes * 8) as u32
}

/// Per-lane bit width of a bit-punned vector.
fn lane_bits(shape: &VecShape) -> u32 {
    image_bits(shape) / shape.lanes
}

/// Whether the lanes are floating (bitcast through the integer image).
fn float_lanes(shape: &VecShape) -> bool {
    matches!(&shape.elem_mir, mir::Ty::Float(_))
}

impl<'s, 'i> Lowerer<'s, 'i> {
    pub(crate) fn vector_shape(&self, ty: &fe::parse::Ty) -> VecShape {
        let fe::parse::Ty::Vector { elem, len } = ty.unqualified() else {
            panic!("vector shape of a non-vector type");
        };
        VecShape {
            elem_fe: (**elem).clone(),
            elem_mir: lower_ty(elem, self.sema, self.interner),
            lanes: *len,
            bytes: self.sema.type_size(ty, self.interner).expect("vector has a size"),
        }
    }

    /// The image integer of a bit-punned vector: its MIR type, widened
    /// to 64 bits for lane extraction (vectors are unsigned bit
    /// containers, so always zero-extended).
    fn vec_image(&self, func: &mut mir::Function, frame: &mut Frame, v: mir::Value, shape: &VecShape) -> mir::Value {
        let bits = image_bits(shape);
        if bits == 64 {
            return v;
        }
        self.emit(func, frame, mir::InstKind::Zext(v), mir::Ty::int(64))
    }

    /// Extract lane `i` of a bit-punned integer vector, sign- or
    /// zero-extended to 64 bits for lane-width arithmetic. A single
    /// 64-bit lane is the value itself.
    fn vec_split_int(&self, func: &mut mir::Function, frame: &mut Frame, v: mir::Value, bits: u32, signed: bool) -> Vec<mir::Value> {
        if bits == 64 {
            return vec![v];
        }
        let mut lanes = Vec::new();
        for i in 0..(64 / bits as usize) {
            let shifted = if i == 0 {
                v
            } else {
                let amount = self.iconst(func, 64, (i as u32 * bits) as i128);
                self.emit(func, frame, mir::InstKind::LShr(v, amount), mir::Ty::int(64))
            };
            let truncated = self.emit(func, frame, mir::InstKind::Trunc(shifted), mir::Ty::int(bits));
            let widened = if signed {
                self.emit(func, frame, mir::InstKind::Sext(truncated), mir::Ty::int(64))
            } else {
                self.emit(func, frame, mir::InstKind::Zext(truncated), mir::Ty::int(64))
            };
            lanes.push(widened);
        }
        lanes
    }

    /// Extract lane `i` of a bit-punned floating vector: the image's
    /// integer lane bitcast to the element float.
    fn vec_split_float(&self, func: &mut mir::Function, frame: &mut Frame, v: mir::Value, shape: &VecShape) -> Vec<mir::Value> {
        let bits = lane_bits(shape);
        let image = self.vec_image(func, frame, v, shape);
        let mut lanes = Vec::new();
        for i in 0..shape.lanes as usize {
            let raw = if i == 0 && bits == 64 {
                image
            } else {
                let shifted = if i == 0 {
                    image
                } else {
                    let amount = self.iconst(func, 64, (i as u32 * bits) as i128);
                    self.emit(func, frame, mir::InstKind::LShr(image, amount), mir::Ty::int(64))
                };
                self.emit(func, frame, mir::InstKind::Trunc(shifted), mir::Ty::int(bits))
            };
            lanes.push(self.emit(func, frame, mir::InstKind::Bitcast(raw), shape.elem_mir.clone()));
        }
        lanes
    }

    /// Rejoin per-lane values (each holding a lane's value in its low
    /// bits) into the integer image of a bit-punned vector. A single
    /// 64-bit lane is the value itself.
    fn vec_join_int(&self, func: &mut mir::Function, frame: &mut Frame, lanes: &[mir::Value], bits: u32, image: u32) -> mir::Value {
        let mut acc: Option<mir::Value> = None;
        for (i, lane) in lanes.iter().enumerate() {
            let truncated = if bits == 64 { *lane } else { self.emit(func, frame, mir::InstKind::Trunc(*lane), mir::Ty::int(bits)) };
            let widened = self.emit(func, frame, mir::InstKind::Zext(truncated), mir::Ty::int(64));
            let placed = if i == 0 {
                widened
            } else {
                let amount = self.iconst(func, 64, (i as u32 * bits) as i128);
                self.emit(func, frame, mir::InstKind::Shl(widened, amount), mir::Ty::int(64))
            };
            acc = Some(match acc {
                None => placed,
                Some(prev) => self.emit(func, frame, mir::InstKind::Or(prev, placed), mir::Ty::int(64)),
            });
        }
        let packed = acc.expect("at least one lane");
        if image == 64 {
            packed
        } else {
            self.emit(func, frame, mir::InstKind::Trunc(packed), mir::Ty::int(image))
        }
    }

    /// Rejoin per-lane float values by bitcasting each through its
    /// integer lane.
    fn vec_join_float(&self, func: &mut mir::Function, frame: &mut Frame, lanes: &[mir::Value], shape: &VecShape) -> mir::Value {
        let bits = lane_bits(shape);
        let mut ints = Vec::new();
        for lane in lanes {
            let raw = self.emit(func, frame, mir::InstKind::Bitcast(*lane), mir::Ty::int(bits));
            ints.push(raw);
        }
        self.vec_join_int(func, frame, &ints, bits, image_bits(shape))
    }

    /// Broadcast a scalar into every lane of a bit-punned vector.
    fn vec_splat(&self, func: &mut mir::Function, frame: &mut Frame, scalar: mir::Value, shape: &VecShape) -> mir::Value {
        let bits = lane_bits(shape);
        if float_lanes(shape) {
            let raw = self.emit(func, frame, mir::InstKind::Bitcast(scalar), mir::Ty::int(bits));
            let lanes = vec![raw; shape.lanes as usize];
            return self.vec_join_int(func, frame, &lanes, bits, image_bits(shape));
        }
        let lanes = vec![scalar; shape.lanes as usize];
        self.vec_join_int(func, frame, &lanes, bits, image_bits(shape))
    }

    /// One lane-wise integer operation on normalized (64-bit widened)
    /// lanes, computed at the lane's own width.
    fn vec_lane_op(&self, func: &mut mir::Function, frame: &mut Frame, op: fe::parse::BinaryOp, a: mir::Value, b: mir::Value, bits: u32, signed: bool) -> mir::Value {
        use fe::parse::BinaryOp::*;
        let lane_ty = mir::Ty::int(bits);
        let (a, b) = if bits == 64 {
            (a, b)
        } else {
            (
                self.emit(func, frame, mir::InstKind::Trunc(a), lane_ty.clone()),
                self.emit(func, frame, mir::InstKind::Trunc(b), lane_ty.clone()),
            )
        };
        let kind = match op {
            Add => mir::InstKind::Add(a, b),
            Sub => mir::InstKind::Sub(a, b),
            Mul => mir::InstKind::Mul(a, b),
            Div => {
                if signed {
                    mir::InstKind::Sdiv(a, b)
                } else {
                    mir::InstKind::Udiv(a, b)
                }
            }
            Rem => {
                if signed {
                    mir::InstKind::Srem(a, b)
                } else {
                    mir::InstKind::Urem(a, b)
                }
            }
            Shl => mir::InstKind::Shl(a, b),
            Shr => {
                if signed {
                    mir::InstKind::AShr(a, b)
                } else {
                    mir::InstKind::LShr(a, b)
                }
            }
            _ => panic!("non-arithmetic lane operation"),
        };
        self.emit(func, frame, kind, lane_ty)
    }

    /// One lane-wise integer comparison on normalized lanes: the mask
    /// lane is 0/`-1` at the lane's own width.
    fn vec_lane_cmp(&self, func: &mut mir::Function, frame: &mut Frame, op: fe::parse::BinaryOp, a: mir::Value, b: mir::Value, bits: u32, signed: bool) -> mir::Value {
        use fe::parse::BinaryOp::*;
        let pred = match (op, signed) {
            (Lt, true) => mir::IcmpPred::Slt,
            (Le, true) => mir::IcmpPred::Sle,
            (Gt, true) => mir::IcmpPred::Sgt,
            (Ge, true) => mir::IcmpPred::Sge,
            (Eq, _) => mir::IcmpPred::Eq,
            (Ne, _) => mir::IcmpPred::Ne,
            (Lt, false) => mir::IcmpPred::Ult,
            (Le, false) => mir::IcmpPred::Ule,
            (Gt, false) => mir::IcmpPred::Ugt,
            (Ge, false) => mir::IcmpPred::Uge,
            _ => panic!("non-comparison lane operation"),
        };
        let lane_ty = mir::Ty::int(bits);
        let (a, b) = if bits == 64 {
            (a, b)
        } else {
            (
                self.emit(func, frame, mir::InstKind::Trunc(a), lane_ty.clone()),
                self.emit(func, frame, mir::InstKind::Trunc(b), lane_ty.clone()),
            )
        };
        let test = self.emit(func, frame, mir::InstKind::Icmp { pred, lhs: a, rhs: b }, mir::Ty::Bool);
        // 0/1 at one byte sign-extends to the lane's 0/`-1` mask.
        self.emit(func, frame, mir::InstKind::Sext(test), lane_ty)
    }

    /// One lane-wise float operation.
    fn vec_lane_fop(&self, func: &mut mir::Function, frame: &mut Frame, op: fe::parse::BinaryOp, a: mir::Value, b: mir::Value, elem: &mir::Ty) -> mir::Value {
        use fe::parse::BinaryOp::*;
        let kind = match op {
            Add => mir::InstKind::FAdd(a, b),
            Sub => mir::InstKind::FSub(a, b),
            Mul => mir::InstKind::FMul(a, b),
            Div => mir::InstKind::FDiv(a, b),
            _ => panic!("non-float lane operation"),
        };
        self.emit(func, frame, kind, elem.clone())
    }

    /// One lane-wise float comparison: the mask lane is 0/`-1` at the
    /// lane's integer width.
    fn vec_lane_fcmp(&self, func: &mut mir::Function, frame: &mut Frame, op: fe::parse::BinaryOp, a: mir::Value, b: mir::Value, bits: u32) -> mir::Value {
        use fe::parse::BinaryOp::*;
        let pred = match op {
            Lt => mir::FcmpPred::Lt,
            Le => mir::FcmpPred::Le,
            Gt => mir::FcmpPred::Gt,
            Ge => mir::FcmpPred::Ge,
            Eq => mir::FcmpPred::Eq,
            Ne => mir::FcmpPred::Ne,
            _ => panic!("non-comparison lane operation"),
        };
        let test = self.emit(func, frame, mir::InstKind::Fcmp { pred, lhs: a, rhs: b }, mir::Ty::Bool);
        self.emit(func, frame, mir::InstKind::Sext(test), mir::Ty::int(bits))
    }

    /// Lower a binary operation with at least one vector operand
    /// (checked: same-shaped vectors, or a scalar splatting in).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn vector_binary(
        &self,
        module: &mut mir::ModuleBuilder,
        func: &mut mir::Function,
        frame: &mut Frame,
        funcs: &HashMap<fe::sema::DeclId, mir::Func>,
        externs: &HashMap<fe::sema::DeclId, mir::Extern>,
        globals: &HashMap<fe::sema::DeclId, mir::Global>,
        op: fe::parse::BinaryOp,
        lhs: &CheckedExpr<'_>,
        rhs: &CheckedExpr<'_>,
        mir_ty: &mir::Ty,
    ) -> mir::Value {
        use fe::parse::BinaryOp::*;
        let shape = self.vector_shape(if fe::check::is_vector(&lhs.ty) { &lhs.ty } else { &rhs.ty });
        let comparison = matches!(op, Lt | Le | Gt | Ge | Eq | Ne);
        // Exactly one side is a vector (the checker enforces the
        // shapes); the scalar side splats, from either position.
        let (x, y) = if fe::check::is_vector(&lhs.ty) {
            let x = self.expr_value(module, func, frame, funcs, externs, globals, lhs);
            let y = if fe::check::is_vector(&rhs.ty) {
                self.expr_value(module, func, frame, funcs, externs, globals, rhs)
            } else {
                let scalar = self.expr_value(module, func, frame, funcs, externs, globals, rhs);
                self.vector_splat(func, frame, scalar, &shape)
            };
            (x, y)
        } else {
            let scalar = self.expr_value(module, func, frame, funcs, externs, globals, lhs);
            let y = self.expr_value(module, func, frame, funcs, externs, globals, rhs);
            (self.vector_splat(func, frame, scalar, &shape), y)
        };
        if shape.bytes < 16 {
            return self.vec_image_binary(func, frame, op, x, y, &shape, mir_ty, comparison);
        }
        // First-class MIR vector: shape-generic instructions.
        let float = float_lanes(&shape);
        if comparison {
            let kind = if float {
                let pred = match op {
                    Lt => mir::FcmpPred::Lt,
                    Le => mir::FcmpPred::Le,
                    Gt => mir::FcmpPred::Gt,
                    Ge => mir::FcmpPred::Ge,
                    Eq => mir::FcmpPred::Eq,
                    Ne => mir::FcmpPred::Ne,
                    _ => unreachable!("comparison selected above"),
                };
                mir::InstKind::Fcmp { pred, lhs: x, rhs: y }
            } else {
                let pred = match (op, is_signed(&shape.elem_fe, self.sema)) {
                    (Lt, true) => mir::IcmpPred::Slt,
                    (Le, true) => mir::IcmpPred::Sle,
                    (Gt, true) => mir::IcmpPred::Sgt,
                    (Ge, true) => mir::IcmpPred::Sge,
                    (Eq, _) => mir::IcmpPred::Eq,
                    (Ne, _) => mir::IcmpPred::Ne,
                    (Lt, false) => mir::IcmpPred::Ult,
                    (Le, false) => mir::IcmpPred::Ule,
                    (Gt, false) => mir::IcmpPred::Ugt,
                    (Ge, false) => mir::IcmpPred::Uge,
                    _ => unreachable!("comparison selected above"),
                };
                mir::InstKind::Icmp { pred, lhs: x, rhs: y }
            };
            // The mask IS the value: lanes hold 0/`-1`, no bool
            // conversion (the checker types the result as the integer
            // mask vector).
            return self.emit(func, frame, kind, mir_ty.clone());
        }
        if matches!(op, Div | Rem) && !float {
            // Cranelift has no packed integer division: expand per
            // lane through ExtractLane and a scratch slot.
            return self.vector_int_divrem(func, frame, op, x, y, &shape, mir_ty);
        }
        let kind = match (op, float) {
            (Add, true) => mir::InstKind::FAdd(x, y),
            (Sub, true) => mir::InstKind::FSub(x, y),
            (Mul, true) => mir::InstKind::FMul(x, y),
            (Div, true) => mir::InstKind::FDiv(x, y),
            (Add, false) => mir::InstKind::Add(x, y),
            (Sub, false) => mir::InstKind::Sub(x, y),
            (Mul, false) => mir::InstKind::Mul(x, y),
            (Div, false) => {
                if is_signed(&shape.elem_fe, self.sema) {
                    mir::InstKind::Sdiv(x, y)
                } else {
                    mir::InstKind::Udiv(x, y)
                }
            }
            (Rem, false) => {
                if is_signed(&shape.elem_fe, self.sema) {
                    mir::InstKind::Srem(x, y)
                } else {
                    mir::InstKind::Urem(x, y)
                }
            }
            (Shl, _) => mir::InstKind::Shl(x, y),
            (Shr, _) => {
                if is_signed(&shape.elem_fe, self.sema) {
                    mir::InstKind::AShr(x, y)
                } else {
                    mir::InstKind::LShr(x, y)
                }
            }
            (BitAnd, _) => mir::InstKind::And(x, y),
            (BitXor, _) => mir::InstKind::Xor(x, y),
            (BitOr, _) => mir::InstKind::Or(x, y),
            _ => panic!("unsupported vector binary operation"),
        };
        self.emit(func, frame, kind, mir_ty.clone())
    }

    /// A binary operation on a bit-punned (2/4/8-byte) vector image.
    /// Bitwise operators are lane-exact as they stand; everything else
    /// expands per lane, bitcasting float lanes through the image.
    fn vec_image_binary(&self, func: &mut mir::Function, frame: &mut Frame, op: fe::parse::BinaryOp, x: mir::Value, y: mir::Value, shape: &VecShape, mir_ty: &mir::Ty, comparison: bool) -> mir::Value {
        use fe::parse::BinaryOp::*;
        let image = image_bits(shape);
        match (op, float_lanes(shape)) {
            (BitAnd, false) => return self.emit(func, frame, mir::InstKind::And(x, y), mir_ty.clone()),
            (BitOr, false) => return self.emit(func, frame, mir::InstKind::Or(x, y), mir_ty.clone()),
            (BitXor, false) => return self.emit(func, frame, mir::InstKind::Xor(x, y), mir_ty.clone()),
            _ => {}
        }
        let bits = lane_bits(shape);
        let (a_lanes, b_lanes) = if float_lanes(shape) {
            (self.vec_split_float(func, frame, x, shape), self.vec_split_float(func, frame, y, shape))
        } else {
            let signed = is_signed(&shape.elem_fe, self.sema);
            let image_x = self.vec_image(func, frame, x, shape);
            let image_y = self.vec_image(func, frame, y, shape);
            (self.vec_split_int(func, frame, image_x, bits, signed), self.vec_split_int(func, frame, image_y, bits, signed))
        };
        let lanes: Vec<mir::Value> = a_lanes
            .iter()
            .zip(&b_lanes)
            .map(|(a, b)| {
                if comparison {
                    if float_lanes(shape) {
                        self.vec_lane_fcmp(func, frame, op, *a, *b, bits)
                    } else {
                        self.vec_lane_cmp(func, frame, op, *a, *b, bits, is_signed(&shape.elem_fe, self.sema))
                    }
                } else if float_lanes(shape) {
                    self.vec_lane_fop(func, frame, op, *a, *b, &shape.elem_mir)
                } else {
                    self.vec_lane_op(func, frame, op, *a, *b, bits, is_signed(&shape.elem_fe, self.sema))
                }
            })
            .collect();
        if comparison || !float_lanes(shape) {
            self.vec_join_int(func, frame, &lanes, bits, image)
        } else {
            self.vec_join_float(func, frame, &lanes, shape)
        }
    }

    /// Packed integer `/`/`%` on a first-class vector, expanded
    /// lane-wise: extract both operands' lanes, divide as scalars
    /// (keeping C's zero-divisor fault), and reassemble through a
    /// scratch slot.
    #[allow(clippy::too_many_arguments)]
    fn vector_int_divrem(&self, func: &mut mir::Function, frame: &mut Frame, op: fe::parse::BinaryOp, x: mir::Value, y: mir::Value, shape: &VecShape, mir_ty: &mir::Ty) -> mir::Value {
        let signed = is_signed(&shape.elem_fe, self.sema);
        let slot = self.alloca(func, frame, mir_ty.clone());
        for lane in 0..shape.lanes {
            let a = self.emit(func, frame, mir::InstKind::ExtractLane { vec: x, lane: lane as u8 }, shape.elem_mir.clone());
            let b = self.emit(func, frame, mir::InstKind::ExtractLane { vec: y, lane: lane as u8 }, shape.elem_mir.clone());
            let kind = match (op, signed) {
                (fe::parse::BinaryOp::Div, true) => mir::InstKind::Sdiv(a, b),
                (fe::parse::BinaryOp::Div, false) => mir::InstKind::Udiv(a, b),
                (fe::parse::BinaryOp::Rem, true) => mir::InstKind::Srem(a, b),
                (fe::parse::BinaryOp::Rem, false) => mir::InstKind::Urem(a, b),
                _ => unreachable!("division or remainder selected above"),
            };
            let q = self.emit(func, frame, kind, shape.elem_mir.clone());
            let idx = self.iconst(func, 64, i128::from(lane));
            let addr = self.scaled_index_addr(func, frame, slot, idx, &fe::parse::Ty::LongLong, shape.elem_mir.size_bytes(), shape.elem_mir.clone()).0;
            self.store(func, frame, addr, q, false);
        }
        self.load(func, frame, slot, mir_ty.clone(), false)
    }

    /// Broadcast a scalar operand across a vector: a first-class MIR
    /// `Splat`, or the bit-punned image with the scalar in every lane.
    pub(crate) fn vector_splat(&self, func: &mut mir::Function, frame: &mut Frame, scalar: mir::Value, shape: &VecShape) -> mir::Value {
        if shape.bytes < 16 {
            return self.vec_splat(func, frame, scalar, shape);
        }
        self.emit(func, frame, mir::InstKind::Splat(scalar), mir::Ty::Vector {
            elem: Box::new(shape.elem_mir.clone()),
            lanes: shape.lanes,
        })
    }

    /// Lane-wise unary minus: float lanes negate in place, integer
    /// lanes subtract from zero (per lane for bit-punned shapes —
    /// a whole-image negation is lane-exact only at one lane).
    pub(crate) fn vector_neg(&self, func: &mut mir::Function, frame: &mut Frame, value: mir::Value, shape: &VecShape, mir_ty: &mir::Ty) -> mir::Value {
        if shape.bytes < 16 {
            if float_lanes(shape) {
                let lanes = self.vec_split_float(func, frame, value, shape);
                let negated: Vec<mir::Value> = lanes
                    .iter()
                    .map(|lane| self.emit(func, frame, mir::InstKind::FNeg(*lane), shape.elem_mir.clone()))
                    .collect();
                return self.vec_join_float(func, frame, &negated, shape);
            }
            let bits = lane_bits(shape);
            let signed = is_signed(&shape.elem_fe, self.sema);
            let image = self.vec_image(func, frame, value, shape);
            let lanes = self.vec_split_int(func, frame, image, bits, signed);
            let zero = self.iconst(func, bits, 0);
            let negated: Vec<mir::Value> = lanes
                .iter()
                .map(|lane| {
                    let truncated = if bits == 64 {
                        *lane
                    } else {
                        self.emit(func, frame, mir::InstKind::Trunc(*lane), mir::Ty::int(bits))
                    };
                    self.emit(func, frame, mir::InstKind::Sub(zero, truncated), mir::Ty::int(bits))
                })
                .collect();
            return self.vec_join_int(func, frame, &negated, bits, image_bits(shape));
        }
        if float_lanes(shape) {
            return self.emit(func, frame, mir::InstKind::FNeg(value), mir_ty.clone());
        }
        let zero_const = self.iconst(func, 32, 0);
        let zero = self.vector_splat(func, frame, zero_const, shape);
        self.emit(func, frame, mir::InstKind::Sub(zero, value), mir_ty.clone())
    }

    /// Lane-wise bitwise complement: XOR against an all-ones vector
    /// (integer lanes only — the checker rejects `~` on float vectors).
    /// The bit-punned image needs nothing special (a whole-image XOR is
    /// lane-exact); the first-class vector splats the element's `-1`.
    pub(crate) fn vector_bitnot(&self, func: &mut mir::Function, frame: &mut Frame, value: mir::Value, shape: &VecShape, mir_ty: &mir::Ty) -> mir::Value {
        if shape.bytes < 16 {
            return self.emit(func, frame, mir::InstKind::Not(value), mir_ty.clone());
        }
        let elem_bits = (shape.elem_mir.size_bytes() * 8) as u32;
        let ones_const = self.iconst(func, elem_bits, -1);
        let ones = self.vector_splat(func, frame, ones_const, shape);
        self.emit(func, frame, mir::InstKind::Xor(value, ones), mir_ty.clone())
    }

    /// The address of lane `index` of a vector object: lanes live
    /// consecutively at `base + i * elem_size`, for every shape.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn vector_lane_addr(&self, func: &mut mir::Function, frame: &mut Frame, base: mir::Value, index: mir::Value, index_ty: &fe::parse::Ty, shape: &VecShape) -> mir::Value {
        self.scaled_index_addr(func, frame, base, index, index_ty, shape.elem_mir.size_bytes(), shape.elem_mir.clone()).0
    }
}
