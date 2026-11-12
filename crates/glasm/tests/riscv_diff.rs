//! Differential tests against GNU as (riscv64-elf-as, binutils 2.45).
//! Each case is assembled by both assemblers; .text bytes and the
//! relocation tables must match exactly. Skipped quietly when the
//! reference assembler is not installed.

mod common;

use common::run;

#[test]
fn riscv_basic_integer() {
    run(
        glasm::Arch::Riscv64,
        "riscv64-elf-as",
        "rv64imac_zicsr_zifencei",
        &[
            (
                "regs",
                r"
        add a0, a1, a2
        add x10, x11, x12
        sub t0, s1, s11
        slt a3, a4, a5
        sltu a6, a7, s2
        xor s3, s4, s5
        or s6, s7, s8
        and s9, s10, t6
        addw a0, a1, a2
        subw a0, a1, a2
        sllw a0, a1, a2
        srlw a0, a1, a2
        sraw a0, a1, a2
        sll t1, t2, t3
        ",
            ),
            (
                "imm_ops",
                r"
        addi a0, a1, -1
        addi a0, a1, 2047
        addi a0, a1, -2048
        slti a0, a1, 42
        sltiu a0, a1, -1
        xori a0, a1, 2047
        ori a0, a1, 31
        andi a0, a1, -32
        addiw a0, a1, 2047
        slli a0, a1, 3
        srli a0, a1, 12
        srai a0, a1, 31
        slliw a0, a1, 1
        srliw a0, a1, 5
        sraiw a0, a1, 17
        lui a0, 0x12345
        lui a0, 0xfffff
        ",
            ),
            (
                "loads_stores",
                r"
        lb a0, 0(a1)
        lh a0, 2(a1)
        lw a0, 4(a1)
        lbu a0, 1(a1)
        lhu a0, 2(a1)
        ld a0, 8(a1)
        lwu a0, 4(a1)
        sb a0, 0(a1)
        sh a0, 2(a1)
        sw a0, 4(a1)
        sd a0, 8(a1)
        lw a0, -2048(sp)
        sw a0, 2047(gp)
        ld a0, 100(sp)
        sd a0, 16(sp)
        lb a0, 1(s2)
        sd t0, 24(s3)
        ",
            ),
            (
                "jumps_branches_near",
                r"
1:      beq a0, a1, 2f
        bne a0, a1, 2f
        blt a0, a1, 2f
        bge a0, a1, 2f
        bltu a0, a1, 2f
        bgeu a0, a1, 2f
        beqz a0, 2f
        bnez a0, 2f
        blez a0, 2f
        bgez a0, 2f
        bltz a0, 2f
        bgtz a0, 2f
        j 2f
        jal ra, 2f
        jal a2, 2f
        ret
        jr a1
        jalr a0, a1, 8
        jalr a0, 8(a1)
        jalr a1
2:      nop
        ",
            ),
        ],
    );
}

#[test]
fn riscv_pseudo_and_compressed() {
    run(
        glasm::Arch::Riscv64,
        "riscv64-elf-as",
        "rv64imac_zicsr_zifencei",
        &[
            (
                "pseudo",
                r"
        mv a0, a1
        not a0, a1
        neg a0, a1
        negw a0, a1
        seqz a0, a1
        snez a0, a1
        sltz a0, a1
        sgtz a0, a1
        bgt a0, a1, 1f
        ble a0, a1, 1f
        bgtu a0, a1, 1f
        bleu a0, a1, 1f
1:      nop
        ecall
        ebreak
        fence
        fence.i
        fence iorw, iorw
        fence rw, rw
        ",
            ),
            (
                "compressed",
                r"
        addi a0, a0, 5
        addi a0, zero, 5
        addi a0, a0, 0
        addi zero, zero, 0
        addi a0, sp, 64
        addi s0, sp, 16
        add a0, a0, a1
        add a0, zero, a1
        sub a0, a0, a1
        and a0, a0, a1
        or a0, a0, a1
        xor a0, a0, a1
        addw a0, a0, a1
        subw a0, a0, a1
        slli a0, a0, 3
        andi s0, s0, 15
        andi a2, a2, 15
        lw a0, 4(sp)
        ld a0, 8(sp)
        sw a0, 4(sp)
        sd a0, 8(sp)
        lw a0, 0(s1)
        ld a0, 8(s1)
        sw a0, 0(s1)
        sd a0, 8(s1)
        lw a0, 12(s2)
        sd a0, 16(s2)
        ",
            ),
            (
                "li_values",
                r"
        li a0, 0
        li a0, 1
        li a0, -1
        li a0, 31
        li a0, -32
        li a0, 32
        li a0, -33
        li a0, 127
        li a0, -128
        li a0, 128
        li a0, -129
        li a0, 2047
        li a0, -2048
        li a0, 2048
        li a0, -2049
        li a0, 4095
        li a0, 4096
        li a0, 0x7fffffff
        li a0, -0x80000000
        li a0, 0x80000000
        li a0, 0xffffffff
        li a0, 0x100000000
        li a0, 0xdeadbeef
        li a0, 0x123456789
        li a0, 0x123456789abcdef
        li a0, -0x123456789abcdef
        li a1, 0x7ff
        li s3, 0x800
        li t5, 0xfffffffffffff7ff
        ",
            ),
        ],
    );
}

#[test]
fn riscv_relocations() {
    run(
        glasm::Arch::Riscv64,
        "riscv64-elf-as",
        "rv64imac_zicsr_zifencei",
        &[
            (
                "hi_lo_local",
                r"
        .data
sym:    .word 5
        .text
        lui a2, %hi(sym)
        addi a2, a2, %lo(sym)
        lw a3, %lo(sym)(a2)
        sw a0, %lo(sym)(a2)
        lui a4, %hi(sym+4)
        addi a4, a4, %lo(sym+4)
        ",
            ),
            (
                "pcrel_pairs",
                r"
        .data
sym:    .word 7
        .text
        la a0, sym
        la a1, sym+8
        auipc a2, %pcrel_hi(sym)
        addi a2, a2, %pcrel_lo(1f)
1:      lw a0, sym
        sw a0, sym, t2
        sw a1, sym, t3
        ",
            ),
            (
                "call_tail_extern",
                r"
        .extern foo
        .extern bar
        .text
        call foo
        tail bar
        call bar
        jal ra, foo
        beq a0, a1, foo
        ",
            ),
            (
                "branch_relocs",
                r"
        .extern ext
        .text
1:      beq a0, a1, ext
        jal ra, ext
        j ext
        la a0, ext
        ",
            ),
        ],
    );
}

#[test]
fn riscv_branch_relaxation() {
    // A branch beyond the compressed/4-byte windows forces the far form.
    run(
        glasm::Arch::Riscv64,
        "riscv64-elf-as",
        "rv64imac_zicsr_zifencei",
        &[
            (
                "far_branch",
                r"
        .text
1:      beq a0, a1, 2f
        bnez a0, 2f
        j 2f
        .space 9000
2:      nop
        ",
            ),
            (
                "mid_branch",
                r"
        .text
1:      beq a0, a1, 2f
        .space 400
2:      nop
        ",
            ),
            (
                "far_backwards",
                r"
        .text
2:      nop
        .space 9000
1:      beq a0, a1, 2b
        bgez a0, 2b
        ",
            ),
        ],
    );
}

#[test]
fn riscv32_basics() {
    // RV32: `li` truncates to 32 bits, W-form mnemonics are rejected,
    // and relaxation/compression behave like the rv64 paths.
    run(
        glasm::Arch::Riscv32,
        "riscv32-elf-as",
        "rv32imac_zicsr_zifencei",
        &[
            (
                "integer",
                r"
        addi a0, a1, -1
        add a0, a1, a2
        sltiu a6, a7, -1
        slli a0, a1, 3
        lui a0, 0x12345
        lw a0, 4(sp)
        sw a0, 8(sp)
        ",
            ),
            (
                "li_truncation",
                r"
        li a0, 0
        li a0, -1
        li a0, 2047
        li a0, 2048
        li a0, -2049
        li a0, 4096
        li a0, 0x7fffffff
        li a0, -0x80000000
        li a0, 0x80000000
        li a0, 0xffffffff
        ",
            ),
        ],
    );
}

#[test]
fn riscv_amos_csrs_system() {
    run(
        glasm::Arch::Riscv64,
        "riscv64-elf-as",
        "rv64imac_zicsr_zifencei",
        &[
            (
                "atomics",
                r"
        lr.w a0, (a1)
        lr.d a0, (a1)
        sc.w a0, a2, (a1)
        sc.d a0, a2, (a1)
        amoswap.w a0, a2, (a1)
        amoswap.d a0, a2, (a1)
        amoadd.w a0, a2, (a1)
        amoadd.d.rl a0, a2, (a1)
        amoxor.w.aq a0, a2, (a1)
        amoand.w.aqrl a0, a2, (a1)
        amoor.w a0, a2, (a1)
        amomin.w a0, a2, (a1)
        amomax.d a0, a2, (a1)
        amominu.w a0, a2, (a1)
        amomaxu.d a0, a2, (a1)
        lr.w.aq a0, (a1)
        sc.w.rl a0, a2, (a1)
        ",
            ),
            (
                "csrs",
                r"
        csrr a0, cycle
        csrw sstatus, a0
        csrs mie, a0
        csrc mip, a0
        csrrw a0, mtvec, a1
        csrrs a0, mstatus, a1
        csrrc a0, satp, a1
        csrrwi a0, mscratch, 5
        csrrsi a0, 0x340, 31
        csrrci a0, 0x341, 0
        csrwi mtval, 1
        csrsi sepc, 2
        csrrw a0, 0x7b0, a1
        csrr a0, 0xc00
        ",
            ),
            (
                "system_data",
                r#"
        .data
msg:    .string "hi"
arr:    .word 1, 2, 3
        .quad 0x123456789abcdef0
        .half 513
        .byte 255
        .text
        ecall
        "#,
            ),
        ],
    );
}
