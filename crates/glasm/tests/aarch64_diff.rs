//! Differential tests against GNU as (native aarch64 `as`).
//! Each case is assembled by both assemblers; .text bytes and the
//! relocation tables must match exactly. Skipped quietly when the
//! reference assembler is not installed.

mod common;

use common::run;

#[test]
fn aarch64_basic_integer() {
    run(
        glasm::Arch::Aarch64,
        "as",
        "",
        &[
            (
                "regs_moves",
                r"
        add x0, x1, x2
        add w3, w4, w5
        sub x6, x7, x8
        adds x9, x10, x11
        subs x12, x13, x14
        mov x0, x1
        mov w2, w3
        mov xzr, x4
        mov sp, x5
        mov x6, sp
        mul x7, x8, x9
        mul w10, w11, w12
        mneg x13, x14, x15
        madd x16, x17, x18, x19
        msub x20, x21, x22, x23
        sdiv x24, x25, x26
        udiv x27, x28, x29
        sdiv w0, w1, w2
        udiv w3, w4, w5
        nop
        ",
            ),
            (
                "add_imm",
                r"
        add x0, x1, #0
        add x2, x3, #4095
        add w4, w5, #255
        add x6, x7, #1, lsl #12
        sub x8, x9, #100
        subs x10, x11, #1
        adds x12, x13, #4095
        adds x14, x15, #1, lsl #12
        cmp x16, #100
        cmp w17, #4095
        cmn x18, #1
        cmn w19, #4095
        add x20, x21, #0
        mov x23, #0
        ",
            ),
            (
                "add_reg_shifts",
                r"
        add x0, x1, x2, lsl #3
        add x3, x4, x5, lsr #1
        add x6, x7, x8, asr #2
        add x9, x10, x11, lsl #5
        add w12, w13, w14, lsl #3
        adds x15, x16, x17, lsl #1
        sub x18, x19, x20, asr #10
        subs x21, x22, x23, lsl #63
        add x24, x25, x26, uxtx #1
        add x27, x28, x29, sxtx
        add x0, x1, w2, uxtw #2
        add x3, x4, w5, sxtw
        add w6, w7, w8, uxtw
        cmp x9, x10, lsl #2
        mov x11, x12, lsl #3
        ",
            ),
            (
                "logical_imm",
                r"
        and x0, x1, #1
        and x2, x3, #0xffff
        orr x4, x5, #0xf0f0f0f0f0f0f0f0
        eor x6, x7, #0xffffffff00000000
        ands x8, x9, #0xff
        orr w10, w11, #0xffff0000
        and w12, w13, #0xff00
        tst x14, #0x8000000000000000
        tst w15, #1
        and x16, x17, #0x00000000000000ff
        eor x18, x19, #0x00ff00ff00ff00ff
        ands x20, x21, #0x7fffffffffffffff
        ",
            ),
            (
                "logical_reg",
                r"
        and x0, x1, x2
        orr x3, x4, x5
        eor x6, x7, x8
        ands x9, x10, x11
        bic x12, x13, x14
        orn x15, x16, x17
        eon x18, x19, x20
        and w21, w22, w23
        orr x24, x25, x26, lsl #4
        eor x27, x28, x29, lsr #7
        and x0, x1, x2, asr #3
        orr x3, x4, x5, ror #10
        mvn x6, x7
        mvn w8, w9
        mvn x10, x11, lsl #2
        mvn x12, x13, asr #1
        tst x14, x15
        mov x16, x17
        mov w18, w19
        ",
            ),
            (
                "shifts",
                r"
        lsl x0, x1, #3
        lsr x2, x3, #63
        asr x4, x5, #32
        ror x6, x7, #1
        lsl w8, w9, #31
        lsr w10, w11, #1
        asr w12, w13, #5
        ror w14, w15, #16
        lsl x16, x17, #0
        lsr x18, x19, #0
        sxtb x20, w21
        sxth x22, w23
        sxtw x24, w25
        uxtb x26, w27
        uxth x28, w29
        uxtw w30, w0
        sxtb w1, w2
        uxth w3, w4
        ",
            ),
            (
                "mov_imms",
                r"
        mov x0, #1
        mov x1, #0xffff
        mov x2, #0x10000
        mov x3, #-1
        mov x4, #-256
        mov x5, #0xf0f0f0f0f0f0f0f0
        mov x8, #0xffffffffffffffff
        mov x9, #0xffffffff00000000
        mov x10, #0xff
        mov w11, #1
        mov w12, #0xffff
        mov w13, #-1
        mov w15, #0x80000000
        mov w16, #0xf0000000
        mov x17, #0x8000000000000000
        mov x18, #0x00ffffff
        mov w19, #0x00ffffff
        movz x20, #0x1234
        movz x21, #0x1234, lsl #16
        movz w22, #0x5678
        movk x23, #0xabcd, lsl #32
        movk w24, #0xabcd, lsl #16
        movn x25, #0
        movn x26, #0x1234, lsl #48
        ",
            ),
            (
                "cond_sel",
                r"
        cset x0, ne
        cset w1, lt
        csetm x2, eq
        csetm w3, hi
        cinc x4, x5, ge
        cinv x6, x7, le
        cneg x8, x9, mi
        csel x10, x11, x12, eq
        csinc x13, x14, x15, cs
        csinv x16, x17, x18, pl
        csneg x19, x20, x21, vs
        ",
            ),
            (
                "branches_local",
                r"
1:      b.ne 2f
        b.eq 2f
        b.ge 2f
        b.lt 2f
        b.gt 2f
        b.le 2f
        b.hs 2f
        b.lo 2f
        b.hi 2f
        b.ls 2f
        b.cs 2f
        b.cc 2f
        b.mi 2f
        b.pl 2f
        b.vs 2f
        b.vc 2f
        b 2f
        bl 2f
        cbz x0, 2f
        cbnz x1, 2f
        cbz w2, 2f
        cbnz w3, 2f
        tbz x4, #3, 2f
        tbnz x5, #63, 2f
        tbz w6, #31, 2f
        tbnz w7, #0, 2f
        br x8
        blr x9
        ret
        ret x10
        cbz x11, 3f
3:      nop
2:      nop
        ",
            ),
        ],
    );
}

#[test]
fn aarch64_load_store() {
    run(
        glasm::Arch::Aarch64,
        "as",
        "",
        &[
            (
                "scaled_offset",
                r"
        ldr x0, [x1]
        ldr x2, [x3, #8]
        ldr x4, [x5, #4095*8]
        ldr w6, [x7, #4]
        ldr w8, [x9, #4095*4]
        str x10, [x11, #16]
        str w12, [x13, #0]
        ldrb w14, [x15, #1]
        strb w16, [x17, #0]
        ldrh w18, [x19, #2]
        strh w20, [x21, #2]
        ldrsb x22, [x23]
        ldrsb w24, [x25]
        ldrsh x26, [x27, #2]
        ldrsh w28, [x29]
        ldrsw x30, [x0, #4]
        ldr x1, [x2, #255]
        ",
            ),
            (
                "unscaled_auto",
                r"
        ldr x0, [x1, #-8]
        ldr x2, [x3, #4]
        ldr x4, [x5, #255]
        ldr x6, [x7, #-256]
        str x8, [x9, #-16]
        ldr w10, [x11, #-4]
        ldrb w12, [x13, #-1]
        strh w14, [x15, #-2]
        ldr x16, [x17, #1]
        ldur x18, [x19, #-8]
        stur x20, [x21, #-16]
        ldurb w22, [x23, #5]
        ldursb x24, [x25, #-1]
        ldursh w26, [x27, #7]
        ldursw x28, [x29, #-3]
        sturh w30, [x0, #-2]
        ",
            ),
            (
                "pre_post_index",
                r"
        str x0, [sp, #-16]!
        ldr x1, [sp], #16
        str w2, [x3, #-4]!
        ldr w4, [x5], #4
        ldrb w6, [x7, #1]!
        strb w8, [x9], #-1
        ldrh w10, [x11], #2
        strh w12, [x13, #-2]!
        ldr x14, [x15, #255]!
        str x16, [x17], #-256
        ldrsb x18, [x19], #1
        ",
            ),
            (
                "reg_index",
                r"
        ldr x0, [x1, x2]
        ldr x3, [x4, x5, lsl #3]
        ldr x6, [x7, w8, sxtw]
        ldr w9, [x10, w11, uxtw #2]
        ldr x12, [x13, w14, sxtw #3]
        ldr x15, [x16, x17, sxtx #3]
        str x18, [x19, x20]
        str w21, [x22, w23, uxtw #2]
        ldrb w24, [x25, x26]
        ldrsh w27, [x28, w29, uxtw #1]
        ",
            ),
            (
                "ldp_stp",
                r"
        stp x29, x30, [sp, #-16]!
        ldp x29, x30, [sp], #16
        ldp x0, x1, [x2]
        ldp w3, w4, [x5]
        stp x6, x7, [x8, #16]
        stp w9, w10, [x11, #4]
        ldp x12, x13, [x14, #-16]
        ldp w15, w16, [x17, #-4]
        stp x18, x19, [x20, #504]
        ldp x21, x22, [sp]
        stp w23, w24, [sp, #8]!
        ldp w25, w26, [x27], #8
        ",
            ),
            (
                "atomics_barriers",
                r"
        dmb sy
        dmb ish
        dmb ishst
        dsb sy
        isb
        ldxr x0, [x1]
        ldxr w2, [x3]
        stxr w4, x5, [x6]
        stxr w7, w8, [x9]
        ldar x10, [x11]
        ldar w12, [x13]
        stlr x14, [x15]
        stlr w16, [x17]
        ",
            ),
        ],
    );
}

#[test]
fn aarch64_relocations() {
    run(
        glasm::Arch::Aarch64,
        "as",
        "",
        &[
            (
                "adrp_lo12_local",
                r"
        .data
sym:    .word 5
arr:    .quad 1, 2, 3
        .text
        adrp x0, sym
        add x0, x0, :lo12:sym
        adrp x1, arr+8
        add x1, x1, :lo12:arr+8
        ldr x2, [x1, :lo12:arr]
        ldr w3, [x2, :lo12:sym]
        str x4, [x3, :lo12:arr+16]
        adr x5, sym
        ",
            ),
            (
                "branch_extern",
                r"
        .extern foo
        .extern bar
        .text
        b foo
        bl bar
        b.ne foo
        cbz x0, foo
        cbnz w1, bar
        tbz x2, #5, foo
        tbnz w3, #1, bar
        bl foo
        ",
            ),
            (
                "branch_local",
                r"
        .text
1:      b 2f
        bl 2f
        b.ne 2f
        cbz x0, 2f
        tbz x1, #1, 2f
        adrp x2, 1b
        adr x3, 2f
        nop
2:      nop
        ",
            ),
            (
                "data_words",
                r#"
        .data
msg:    .string "hi"
arr:    .word 1, 2, 3
        .quad 0x123456789abcdef0
        .hword 513
        .byte 255
        .text
        nop
        "#,
            ),
        ],
    );
}

#[test]
fn debug_one() {
    if let Ok(src) = std::env::var("A64DBG_SRC") {
        match glasm::gas::assemble_to_elf(&src, glasm::Arch::Aarch64) {
            Ok(elf) => {
                let out = std::env::var("A64DBG_OUT").unwrap_or_else(|_| "/tmp/a64dbg.o".into());
                std::fs::write(out, elf).unwrap();
            }
            Err(e) => println!("ASM ERR: {e:?}"),
        }
    }
    if let Ok(v) = std::env::var("A64DBG_IMM") {
        let is64 = std::env::var("A64DBG_64").is_ok();
        println!("{v} is64={is64} -> {:?}", glasm::aarch64::encode_logical_imm(u64::from_str_radix(&v, 16).unwrap(), is64));
    }
}

#[test]
fn aarch64_edge_cases() {
    // Forms the compiler's inline-asm wrappers and naked asm emit.
    run(
        glasm::Arch::Aarch64,
        "as",
        "",
        &[
            (
                "wrapper_forms",
                r"
        str x0, [sp, #-16]!
        ldr x0, [sp], #16
        mov x9, x0
        mov x9, #25
        b .Ltarget
        bl .Lfn
.Ltarget:
.Lfn:
        ret
        ",
            ),
            (
                "extended_more",
                r"
        add x0, x1, w2, uxtw
        adds x3, x4, w5, sxtw #2
        subs x6, x7, x8, uxtx
        adds x9, x10, x11, sxtx #2
        sub w12, w13, w14, uxtw #3
        cmp x15, w16, sxtw
        ",
            ),
            (
                "mov_imm_torture",
                r"
        mov x0, #0x8000000000000000
        mov x1, #0x7fffffffffffffff
        mov x2, #0xffffffffffffff00
        mov x3, #0x000000000000ff00
        mov x4, #0xf000000000000000
        mov x5, #0x000000000000000f
        mov x6, #0xf0f0f0f0f0f0f0f0
        mov x7, #0xffff0000ffff0000
        mov x8, #0x0808080808080808
        mov x9, #0x0180018001800180
        mov w10, #0xffff0000
        mov w11, #0x0000ffff
        mov w12, #0xff000000
        mov w13, #0x000000ff
        mov w14, #0xf0f0f0f0
        mov w15, #0x01800180
        mov x16, #-0x80000000
        mov w17, #-0x80000000
        mov x18, #0x80000000
        ",
            ),
            (
                "adrp_got",
                r"
        .extern extsym
        .data
loc:    .word 1
        .text
        adrp x0, :got:extsym
        ldr x0, [x0, :got_lo12:extsym]
        adrp x1, loc
        add x1, x1, :lo12:loc
        ",
            ),
            (
                "bcond_far_local",
                r"
        .text
1:      b.ne 2f
        .space 400000
2:      b 1b
        ",
            ),
        ],
    );
}
