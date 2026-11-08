//! Assembly text printer for debugging (the `-emit-asm` style dump of
//! the native backend). Not a GAS frontend — output is human-readable,
//! with post-allocation registers spelled out.

use crate::isa::TargetIsa;
use crate::machine::{MFunction, Operand, Reg};

pub struct Printer {
    out: String,
}

impl Printer {
    pub fn new() -> Self {
        Printer { out: String::new() }
    }

    pub fn finish(self) -> String {
        self.out
    }

    pub fn function(&mut self, f: &MFunction, isa: &'static dyn TargetIsa) {
        self.out.push_str(&format!("{}{}:\n", if f.local { "" } else { ".globl placeholder " }, f.name));
        let reginfo = isa.reginfo();
        for (bi, block) in f.blocks.iter().enumerate() {
            self.out.push_str(&format!(".L{}_{bi}:\n", f.name));
            for inst in &block.insts {
                let name = if inst.op.is_pseudo() {
                    match inst.op {
                        crate::machine::OpId::MOVE => "mov",
                        crate::machine::OpId::RET => "ret",
                        crate::machine::OpId::JMP => "jmp",
                        _ => "pseudo",
                    }
                    .to_string()
                } else {
                    isa.opcodes().spec(inst.op).name.to_string()
                };
                let ops: Vec<String> = inst.ops.iter().map(|o| operand_text(o, reginfo)).collect();
                let ops_text = if ops.is_empty() { String::new() } else { format!(" {}", ops.join(", ")) };
                let branch_text = inst
                    .branch
                    .as_ref()
                    .map(|br| format!("   ; -> {:?}", br.targets.iter().map(|t| format!(".L{t}")).collect::<Vec<_>>()))
                    .unwrap_or_default();
                self.out.push_str(&format!("    {name}{ops_text}{branch_text}\n"));
            }
        }
        self.out.push('\n');
    }
}

fn operand_text(o: &Operand, reginfo: &crate::isa::RegInfo) -> String {
    match o {
        Operand::Reg(Reg::P(p)) | Operand::FixedReg(Reg::P(p), _) => reginfo.info(*p).name.to_string(),
        Operand::Reg(Reg::P(_) | Reg::V(_)) | Operand::Reg(_) | Operand::FixedReg(_, _) => "?vreg".into(),
        Operand::Imm(i) => format!("${i}"),
        Operand::FrameIndex(i) => format!("%slot{i}"),
        Operand::Mem(m) => {
            let base = m.base.map(reg_text).unwrap_or_else(|| "rip".into());
            let index = m.index.map(|r| format!("+{}", reg_text(r))).unwrap_or_default();
            format!("{}({}{index})", m.disp, base)
        }
        Operand::Sym(s) => format!("{}{:?}", if s.addend < 0 { "-" } else { "" }, s.target),
    }
}

fn reg_text(r: Reg) -> String {
    match r {
        Reg::P(p) => format!("%{}", p.hw),
        Reg::V(v) => format!("%v{}", v.0),
        Reg::FI(i) => format!("%slot{i}"),
        Reg::Spill(i) => format!("%spill{i}"),
    }
}
