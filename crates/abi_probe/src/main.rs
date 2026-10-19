//! ABI conformance prober.
//!
//! Each case lowers a C module through the real pipeline
//! (frontend -> MIR -> Clif object) for one target, links it against a
//! C harness, runs the result (natively or under qemu-user), and diffs
//! stdout. Green across the matrix means our calling conventions match
//! the platform C ABI in both directions.
//!
//! Requirements: `cc`, `zig`, `qemu-aarch64`, `qemu-riscv64`.

use std::path::{Path, PathBuf};
use std::process::Command;

use glcc_lower_cl::{Arch, Os, Target};

const WORKDIR: &str = "/tmp/opencode/abi_probe";

#[derive(Clone, Copy, PartialEq, Eq)]
enum Machine {
    X86_64,
    Aarch64,
    Riscv64,
}

impl Machine {
    fn target(&self) -> Target {
        Target {
            arch: match self {
                Machine::X86_64 => Arch::X86_64,
                Machine::Aarch64 => Arch::Aarch64,
                Machine::Riscv64 => Arch::Riscv64,
            },
            os: Os::Linux,
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Machine::X86_64 => "x86_64",
            Machine::Aarch64 => "aarch64",
            Machine::Riscv64 => "riscv64",
        }
    }

    fn zig_triple(&self) -> &'static str {
        match self {
            Machine::X86_64 => "x86_64-linux-musl",
            Machine::Aarch64 => "aarch64-linux-musl",
            Machine::Riscv64 => "riscv64-linux-musl",
        }
    }

    fn runner(&self) -> Vec<&'static str> {
        match self {
            Machine::X86_64 => vec![],
            Machine::Aarch64 => vec!["qemu-aarch64"],
            Machine::Riscv64 => vec!["qemu-riscv64"],
        }
    }

    fn fe_target(&self) -> glcc_frontend::target::Target {
        match self {
            Machine::X86_64 => glcc_frontend::target::Target::X86_64,
            Machine::Aarch64 => glcc_frontend::target::Target::AARCH64,
            Machine::Riscv64 => glcc_frontend::target::Target::RISCV64,
        }
    }
}

struct Case {
    /// Short id, used for file names.
    name: &'static str,
    /// C sources compiled through OUR pipeline (frontend -> MIR).
    module_src: &'static str,
    /// C harness with `main`, compiled by the platform compiler.
    harness_src: &'static str,
    expected_stdout: &'static str,
    /// extra objects to link (e.g. the x86-64 variadic trampoline).
    needs_glccrt: bool,
    machines: &'static [Machine],
    /// Machines where failure is a TRACKED divergence (documented in
    /// code), not a regression: reported as XFAIL and still green. A
    /// pass there is an XPASS and fails the run, so fixed gaps
    /// announce themselves.
    xfail: &'static [Machine],
}

const ALL: &[Machine] = &[Machine::X86_64, Machine::Aarch64, Machine::Riscv64];

fn cases() -> Vec<Case> {
    vec![
        Case {
            name: "int9",
            module_src: "int sum9(int a,int b,int c,int d,int e,int f,int g,int h,int i){return a+b+c+d+e+f+g+h+i;}\n",
            harness_src: "#include <stdio.h>\nint sum9(int,int,int,int,int,int,int,int,int);\nint main(){printf(\"%d\\n\", sum9(1,2,3,4,5,6,7,8,9)); return 0;}\n",
            expected_stdout: "45\n",
            needs_glccrt: false,
            machines: ALL,
            xfail: &[],
        },
        Case {
            name: "mixed_scalar",
            module_src: "double f(int a, double b, char c, short d, float e, long f){return a + b + c + d + e + f;}\n",
            harness_src: "#include <stdio.h>\ndouble f(int,double,char,short,float,long);\nint main(){printf(\"%f\\n\", f(1, 2.5, 3, 4, 5.5f, 6)); return 0;}\n",
            expected_stdout: "22.000000\n",
            needs_glccrt: false,
            machines: ALL,
            xfail: &[],
        },
        Case {
            name: "sp_P",
            module_src: "struct P{int x;int y;}; int f(struct P p){return p.x*10+p.y;}\n",
            harness_src: "#include <stdio.h>\nstruct P{int x;int y;}; int f(struct P);\nint main(){struct P p = {1,2}; printf(\"%d\\n\", f(p)); return 0;}\n",
            expected_stdout: "12\n",
            needs_glccrt: false,
            machines: ALL,
            xfail: &[],
        },
        Case {
            name: "sp_Q",
            module_src: "struct Q{long a;long b;}; long f(struct Q q){return q.a+q.b;}\n",
            harness_src: "#include <stdio.h>\nstruct Q{long a;long b;}; long f(struct Q);\nint main(){struct Q q = {10,20}; printf(\"%ld\\n\", f(q)); return 0;}\n",
            expected_stdout: "30\n",
            needs_glccrt: false,
            machines: ALL,
            xfail: &[],
        },
        Case {
            name: "sp_R",
            module_src: "struct R{double a;double b;}; double f(struct R r){return r.a+r.b;}\n",
            harness_src: "#include <stdio.h>\nstruct R{double a;double b;}; double f(struct R);\nint main(){struct R r = {1.5,2.5}; printf(\"%f\\n\", f(r)); return 0;}\n",
            expected_stdout: "4.000000\n",
            needs_glccrt: false,
            machines: ALL,
            xfail: &[],
        },
        Case {
            name: "sp_M",
            module_src: "struct M{double d;int i;}; double f(struct M m){return m.d+m.i;}\n",
            harness_src: "#include <stdio.h>\nstruct M{double d;int i;}; double f(struct M);\nint main(){struct M m = {1.5,2}; printf(\"%f\\n\", f(m)); return 0;}\n",
            expected_stdout: "3.500000\n",
            needs_glccrt: false,
            machines: ALL,
            xfail: &[],
        },
        Case {
            name: "sp_S",
            module_src: "struct S{float a;float b;}; float f(struct S s){return s.a+s.b;}\n",
            harness_src: "#include <stdio.h>\nstruct S{float a;float b;}; float f(struct S);\nint main(){struct S s = {1.5f,2.5f}; printf(\"%f\\n\", (double)f(s)); return 0;}\n",
            expected_stdout: "4.000000\n",
            needs_glccrt: false,
            machines: ALL,
            xfail: &[],
        },
        Case {
            name: "sp_T",
            module_src: "struct T{float a;float b;float c;float d;}; float f(struct T t){return t.a+t.b+t.c+t.d;}\n",
            harness_src: "#include <stdio.h>\nstruct T{float a;float b;float c;float d;}; float f(struct T);\nint main(){struct T t = {1.0f,2.0f,3.0f,4.0f}; printf(\"%f\\n\", (double)f(t)); return 0;}\n",
            expected_stdout: "10.000000\n",
            needs_glccrt: false,
            machines: ALL,
            xfail: &[],
        },
        Case {
            name: "sp_U",
            module_src: "struct U{int a;float b;}; float f(struct U u){return u.a+u.b;}\n",
            harness_src: "#include <stdio.h>\nstruct U{int a;float b;}; float f(struct U);\nint main(){struct U u = {1,2.5f}; printf(\"%f\\n\", (double)f(u)); return 0;}\n",
            expected_stdout: "3.500000\n",
            needs_glccrt: false,
            machines: ALL,
            xfail: &[],
        },
        Case {
            name: "sp_Twelve",
            module_src: "struct Twelve{int a[3];}; int f(struct Twelve t){return t.a[0]+t.a[1]+t.a[2];}\n",
            harness_src: "#include <stdio.h>\nstruct Twelve{int a[3];}; int f(struct Twelve);\nint main(){struct Twelve t = {{1,2,3}}; printf(\"%d\\n\", f(t)); return 0;}\n",
            expected_stdout: "6\n",
            needs_glccrt: false,
            machines: ALL,
            xfail: &[],
        },
        Case {
            name: "sr_P",
            module_src: "struct P{int x;int y;}; struct P f(int n){struct P p; p.x = n+1; p.y = n+2; return p;}\n",
            harness_src: "#include <stdio.h>\nstruct P{int x;int y;}; struct P f(int);\nint main(){struct P p = f(10); printf(\"%d %d\\n\", p.x, p.y); return 0;}\n",
            expected_stdout: "11 12\n",
            needs_glccrt: false,
            machines: ALL,
            xfail: &[],
        },
        Case {
            name: "sr_R",
            module_src: "struct R{double a;double b;}; struct R f(double x){struct R r; r.a = x; r.b = x*2.0; return r;}\n",
            harness_src: "#include <stdio.h>\nstruct R{double a;double b;}; struct R f(double);\nint main(){struct R r = f(1.5); printf(\"%f %f\\n\", r.a, r.b); return 0;}\n",
            expected_stdout: "1.500000 3.000000\n",
            needs_glccrt: false,
            machines: ALL,
            xfail: &[],
        },
        Case {
            name: "sr_Big",
            module_src: "struct Big{long a[4];}; struct Big f(long n){struct Big b; b.a[0]=n; b.a[1]=n+1; b.a[2]=n+2; b.a[3]=n+3; return b;}\n",
            harness_src: "#include <stdio.h>\nstruct Big{long a[4];}; struct Big f(long);\nint main(){struct Big b = f(10); printf(\"%ld %ld %ld %ld\\n\", b.a[0], b.a[1], b.a[2], b.a[3]); return 0;}\n",
            expected_stdout: "10 11 12 13\n",
            needs_glccrt: false,
            machines: ALL,
            xfail: &[],
        },
        Case {
            name: "sp_Big",
            module_src: "struct Big{long a[4];}; long f(struct Big b){return b.a[0]+b.a[1]+b.a[2]+b.a[3];}\n",
            harness_src: "#include <stdio.h>\nstruct Big{long a[4];}; long f(struct Big);\nint main(){struct Big b = {{1,2,3,4}}; printf(\"%ld\\n\", f(b)); return 0;}\n",
            expected_stdout: "10\n",
            needs_glccrt: false,
            machines: ALL,
            xfail: &[],
        },
        Case {
            name: "scall_Q",
            module_src: "struct Q{long a;long b;}; long probeQ(struct Q q);\nlong f(struct Q q){return probeQ(q)+1;}\n",
            harness_src: "#include <stdio.h>\nstruct Q{long a;long b;};\nlong probeQ(struct Q q){return q.a*100+q.b;}\nlong f(struct Q);\nint main(){struct Q q = {3,4}; printf(\"%ld\\n\", f(q)); return 0;}\n",
            expected_stdout: "305\n",
            needs_glccrt: false,
            machines: ALL,
            xfail: &[],
        },
        Case {
            name: "scall_R",
            module_src: "struct R{double a;double b;}; double probeR(struct R r);\ndouble f(struct R r){return probeR(r)+1.0;}\n",
            harness_src: "#include <stdio.h>\nstruct R{double a;double b;};\ndouble probeR(struct R r){return r.a*10.0+r.b;}\ndouble f(struct R);\nint main(){struct R r = {1.5,2.5}; printf(\"%f\\n\", f(r)); return 0;}\n",
            expected_stdout: "18.500000\n",
            needs_glccrt: false,
            machines: ALL,
            xfail: &[],
        },
        Case {
            name: "scall_Big",
            module_src: "struct Big{long a[4];}; long probeBig(struct Big b);\nlong f(struct Big b){return probeBig(b)+1;}\n",
            harness_src: "#include <stdio.h>\nstruct Big{long a[4];};\nlong probeBig(struct Big b){return b.a[0]+b.a[1]+b.a[2]+b.a[3];}\nlong f(struct Big);\nint main(){struct Big b = {{1,2,3,4}}; printf(\"%ld\\n\", f(b)); return 0;}\n",
            expected_stdout: "11\n",
            needs_glccrt: false,
            machines: ALL,
            xfail: &[],
        },
        Case {
            name: "ld_param",
            module_src: "long double f(long double x){return x;}\n",
            harness_src: "#include <stdio.h>\nlong double f(long double);\nint main(){long double x = f(1.5L); printf(\"%d\\n\", x == 1.5L); return 0;}\n",
            expected_stdout: "1\n",
            needs_glccrt: false,
            machines: &[Machine::Aarch64, Machine::Riscv64],
            xfail: &[],
        },
        Case {
            name: "ld_self",
            module_src: "long double seen; void f(long double x){seen = x;}\nlong double mk(long double x){seen = x; return x;}\n",
            harness_src: "#include <stdio.h>\nvoid f(long double); long double mk(long double);\nextern long double seen;\nint main(){f(1.5L); int a = seen == 1.5L; long double y = mk(2.5L); int b = seen == 2.5L; (void)y; printf(\"%d %d\\n\", a, b); return 0;}\n",
            expected_stdout: "1 1\n",
            needs_glccrt: false,
            machines: &[Machine::X86_64],
            xfail: &[],
        },
        Case {
            name: "v_sum",
            module_src: "typedef __builtin_va_list va_list;\nint vsum(int n, ...) {\nva_list ap; __builtin_va_start(ap, n);\nint s = 0;\nfor (int i = 0; i < n; i++) s += __builtin_va_arg(ap, int);\n__builtin_va_end(ap);\nreturn s;}\n",
            harness_src: "#include <stdio.h>\nint vsum(int, ...);\nint main(){printf(\"%d %d\\n\", vsum(3, 10, 20, 30), vsum(8, 1, 2, 3, 4, 5, 6, 7, 8)); return 0;}\n",
            expected_stdout: "60 36\n",
            needs_glccrt: false,
            machines: ALL,
            xfail: &[],
        },
        Case {
            name: "v_mix",
            module_src: "typedef __builtin_va_list va_list;\ndouble vmix(int n, ...) {\nva_list ap; __builtin_va_start(ap, n);\ndouble d = __builtin_va_arg(ap, double);\nint i = __builtin_va_arg(ap, int);\ndouble e = __builtin_va_arg(ap, double);\n__builtin_va_end(ap);\n(void)n; return d + i + e;}\n",
            harness_src: "#include <stdio.h>\ndouble vmix(int, ...);\nint main(){printf(\"%f\\n\", vmix(0, 1.5, 7, 2.5)); return 0;}\n",
            expected_stdout: "11.000000\n",
            needs_glccrt: false,
            machines: ALL,
            xfail: &[],
        },
        Case {
            name: "v_copy",
            module_src: "typedef __builtin_va_list va_list;\nint vsum2(int n, ...) {\nva_list ap; __builtin_va_start(ap, n);\nva_list aq; __builtin_va_copy(aq, ap);\nint s = 0;\nfor (int i = 0; i < n; i++) s += __builtin_va_arg(ap, int);\nint t = 0;\nfor (int i = 0; i < n; i++) t += __builtin_va_arg(aq, int);\n__builtin_va_end(ap); __builtin_va_end(aq);\nreturn s * 100 + t;}\n",
            harness_src: "#include <stdio.h>\nint vsum2(int, ...);\nint main(){printf(\"%d\\n\", vsum2(2, 10, 20)); return 0;}\n",
            expected_stdout: "3030\n",
            needs_glccrt: false,
            machines: ALL,
            xfail: &[],
        },
        Case {
            name: "v_printf",
            module_src: "int printf(const char *, ...);\nint callem(int a, double b, const char *s){return printf(\"%d %f %s\\n\", a, b, s);}\n",
            harness_src: "#include <stdio.h>\nint callem(int, double, const char*);\nint main(){int n = callem(42, 1.5, \"hi\"); printf(\"n=%d\\n\", n >= 10); return 0;}\n",
            expected_stdout: "42 1.500000 hi\nn=1\n",
            needs_glccrt: true,
            machines: ALL,
            xfail: &[],
        },
        Case {
            name: "v_self",
            module_src: "typedef __builtin_va_list va_list;\nint vsum(int n, ...) {\nva_list ap; __builtin_va_start(ap, n);\nint s = 0;\nfor (int i = 0; i < n; i++) s += __builtin_va_arg(ap, int);\n__builtin_va_end(ap);\nreturn s;}\nint entry(){return vsum(3, 10, 20, 30);}\nint entry8(){return vsum(8, 1, 2, 3, 4, 5, 6, 7, 8);}\n",
            harness_src: "#include <stdio.h>\nint entry();\nint entry8();\nint main(){printf(\"%d %d\\n\", entry(), entry8()); return 0;}\n",
            expected_stdout: "60 36\n",
            needs_glccrt: false,
            machines: ALL,
            xfail: &[],
        },
        Case {
            name: "v_mixself",
            module_src: "typedef __builtin_va_list va_list;\ndouble vmix(int n, ...) {\nva_list ap; __builtin_va_start(ap, n);\ndouble d = __builtin_va_arg(ap, double);\nint i = __builtin_va_arg(ap, int);\ndouble e = __builtin_va_arg(ap, double);\n__builtin_va_end(ap);\n(void)n; return d + i + e;}\ndouble entry(){return vmix(0, 1.5, 7, 2.5);}\n",
            harness_src: "#include <stdio.h>\ndouble entry();\nint main(){printf(\"%f\\n\", entry()); return 0;}\n",
            expected_stdout: "11.000000\n",
            needs_glccrt: false,
            machines: ALL,
            xfail: &[],
        },
        Case {
            name: "switch",
            module_src: "int f(int x){switch(x){case 1: return 10; case 2: case 3: return 20; default: return 30;}}\nint fall(int x){int s = 0; switch(x){case 1: s += 1; case 2: s += 10; default: s += 100;} return s;}\n",
            harness_src: "#include <stdio.h>\nint f(int);\nint fall(int);\nint main(){printf(\"%d %d %d %d %d\\n\", f(1), f(3), f(9), fall(1), fall(2)); return 0;}\n",
            expected_stdout: "10 20 30 111 110\n",
            needs_glccrt: false,
            machines: ALL,
            xfail: &[],
        },
        Case {
            name: "switch_char",
            // Plain `char` is signed on x86-64 but unsigned on
            // AArch64/RISC-V (matching clang), so the promotion check
            // uses `signed char` for a target-independent expectation.
            module_src: "int f(signed char c){switch(c){case 255: return 1; case -1: return 2; default: return 3;}}\nint g(unsigned char c){switch(c){case 255: return 1; case -1: return 2; default: return 3;}}\n",
            harness_src: "#include <stdio.h>\nint f(signed char);\nint g(unsigned char);\nint main(){printf(\"%d %d\\n\", f(-1), g(255)); return 0;}\n",
            expected_stdout: "2 1\n",
            needs_glccrt: false,
            machines: ALL,
            xfail: &[],
        },
        Case {
            name: "goto",
            module_src: "int loop(int n){int i = 0; int s = 0; again: i++; s += i; if(i < n) goto again; return s;}\nint jump(int x){goto mid; x += 100; mid: return x + 1;}\n",
            harness_src: "#include <stdio.h>\nint loop(int);\nint jump(int);\nint main(){printf(\"%d %d\\n\", loop(10), jump(41)); return 0;}\n",
            expected_stdout: "55 42\n",
            needs_glccrt: false,
            machines: ALL,
            xfail: &[],
        },
        Case {
            name: "duff",
            module_src: "int f(int n){int s = 0; int i = 0; switch(n){case 0: for(i = 0; i < 4; i++){case 2: s += 10; case 3: s += 1;}} return s * 100 + i;}\n",
            harness_src: "#include <stdio.h>\nint f(int);\nint main(){printf(\"%d %d %d\\n\", f(0), f(2), f(3)); return 0;}\n",
            expected_stdout: "4404 4404 3404\n",
            needs_glccrt: false,
            machines: ALL,
            xfail: &[],
        },
        Case {
            name: "list",
            module_src: "struct Node{int v; struct Node *next;};\nint sum(struct Node *p){int s = 0; while(p){s += p->v; p = p->next;} return s;}\nint hop(struct Node *p){return p->next->next->v;}\nint entry(){struct Node n1; struct Node n2; struct Node n3;\nn1.v = 10; n2.v = 20; n3.v = 30;\nn1.next = &n2; n2.next = &n3; n3.next = 0;\nreturn sum(&n1) * 100 + hop(&n1);}\nstruct P{int x; int y;};\nint at(struct P *p, int i){return p[i].x + p[i].y;}\nint entry2(){struct P a[3] = {{1,2},{3,4},{5,6}}; return at(a, 0) + at(a, 1) * 10 + at(a, 2) * 100;}\n",
            harness_src: "#include <stdio.h>\nint entry();\nint entry2();\nint main(){printf(\"%d %d\\n\", entry(), entry2()); return 0;}\n",
            expected_stdout: "6030 1173\n",
            needs_glccrt: false,
            machines: ALL,
            xfail: &[],
        },
        Case {
            name: "union",
            module_src: "union N{int a; int b;};\nint geta(union N u){return u.a;}\nunion N mk(int x){union N u; u.b = x; return u;}\nint entry(){union N u; u.a = 42; union N w = mk(7); return geta(u) + geta(w) * 100;}\nunion DL{double d; long l;};\nlong getl(union DL d){return d.l;}\nint entry2(){union DL d; d.l = 5; return (int)getl(d) + (d.l == 5 ? 1000 : 0);}\n",
            harness_src: "#include <stdio.h>\nint entry();\nint entry2();\nint main(){printf(\"%d %d\\n\", entry(), entry2()); return 0;}\n",
            expected_stdout: "742 1005\n",
            needs_glccrt: false,
            machines: ALL,
            xfail: &[],
        },
        Case {
            name: "union_mixed",
            // `union{float f[3]; short s[3]}` (12B): the upper
            // eightbyte is float-only, so x86-64 crosses it in xmm0
            // while the byte blob sends a GPR. Tracked, not fixed.
            module_src: "union M{float f[3]; short s[3];};\nint check(union M u){return u.f[2] == 3.5f && u.s[0] == 7;}\n",
            harness_src: "#include <stdio.h>\nunion M{float f[3]; short s[3];};\nint check(union M);\nint main(){union M u; u.f[0] = 1.5f; u.f[1] = 2.5f; u.f[2] = 3.5f; u.s[0] = 7; printf(\"%d\\n\", check(u)); return 0;}\n",
            expected_stdout: "1\n",
            needs_glccrt: false,
            machines: ALL,
            xfail: &[Machine::X86_64],
        },
        Case {
            name: "union_fp",
            // All-floating-point unions cross in SIMD registers on
            // x86-64/AAPCS64; the blob sends integer registers.
            // Tracked, not fixed. RISC-V takes GPRs either way.
            // (One member only: overlay means a second store would
            // clobber the checked value on every target alike.)
            module_src: "union DD{double a; double b;};\nint check(union DD u){return u.b == 2.5;}\n",
            harness_src: "#include <stdio.h>\nunion DD{double a; double b;};\nint check(union DD);\nint main(){union DD u; u.b = 2.5; printf(\"%d\\n\", check(u)); return 0;}\n",
            expected_stdout: "1\n",
            needs_glccrt: false,
            machines: ALL,
            xfail: &[Machine::X86_64, Machine::Aarch64],
        },
        Case {
            name: "i128_scalar",
            module_src: "unsigned __int128 ident(unsigned __int128 a){return a;}\n",
            harness_src: "#include <stdio.h>\nunsigned __int128 ident(unsigned __int128);\nint main(){printf(\"%d\\n\", (int)ident(100) + (int)ident(23)); return 0;}\n",
            expected_stdout: "123\n",
            needs_glccrt: false,
            machines: ALL,
            xfail: &[],
        },
        Case {
            name: "bitfield",
            module_src: "struct B{int a:3; int b:5;};\nint get(struct B s){return s.a * 100 + s.b;}\nstruct B mk(int x){struct B s; s.a = x; s.b = x + 1; return s;}\nstruct W{unsigned u:5; long l:40;};\nunsigned getu(struct W w){return w.u;}\nlong getl(struct W w){return w.l;}\nstruct BD{double d; int x:3;};\ndouble getd(struct BD s){return s.d + s.x;}\nstruct WL{long l:40;};\nlong getwl(struct WL w){return w.l;}\n",
            harness_src: "#include <stdio.h>\nstruct B{int a:3; int b:5;};\nint get(struct B);\nstruct B mk(int);\nstruct W{unsigned u:5; long l:40;};\nunsigned getu(struct W);\nlong getl(struct W);\nstruct BD{double d; int x:3;};\ndouble getd(struct BD);\nstruct WL{long l:40;};\nlong getwl(struct WL);\nint main(){struct B b = {1, 2}; printf(\"%d\\n\", get(b));\nstruct B m = mk(2); printf(\"%d %d\\n\", m.a, m.b);\nstruct W w; w.u = 17; w.l = 1000000000L; printf(\"%u %ld\\n\", getu(w), getl(w));\nstruct BD d; d.d = 1.5; d.x = 2; printf(\"%f\\n\", getd(d));\nstruct WL t; t.l = (1L<<39)|5; printf(\"%ld\\n\", getwl(t)); return 0;}\n",
            expected_stdout: "102\n2 3\n17 1000000000\n3.500000\n-549755813883\n",
            needs_glccrt: false,
            machines: ALL,
            xfail: &[],
        },
        Case {
            name: "v_struct",
            module_src: "typedef __builtin_va_list va_list;\nstruct P{int x;int y;};\nint vsum(int n, ...) {\nva_list ap; __builtin_va_start(ap, n);\nint s = 0;\nfor (int i = 0; i < n; i++) { struct P p = __builtin_va_arg(ap, struct P); s += p.x + p.y; }\n__builtin_va_end(ap);\nreturn s;}\nint entry(){return vsum(2, (struct P){1,2}, (struct P){3,4});}\n",
            harness_src: "#include <stdio.h>\nint entry();\nint main(){printf(\"%d\\n\", entry()); return 0;}\n",
            expected_stdout: "10\n",
            needs_glccrt: false,
            machines: ALL,
            xfail: &[],
        },
        Case {
            name: "v_struct_c",
            module_src: "typedef __builtin_va_list va_list;\nstruct P{int x;int y;};\nint vsum(int n, ...) {\nva_list ap; __builtin_va_start(ap, n);\nint s = 0;\nfor (int i = 0; i < n; i++) { struct P p = __builtin_va_arg(ap, struct P); s += p.x + p.y; }\n__builtin_va_end(ap);\nreturn s;}\n",
            harness_src: "#include <stdio.h>\nstruct P{int x;int y;};\nint vsum(int, ...);\nint main(){struct P a = {1,2}; struct P b = {3,4}; printf(\"%d\\n\", vsum(2, a, b)); return 0;}\n",
            expected_stdout: "10\n",
            needs_glccrt: false,
            machines: ALL,
            xfail: &[],
        },
    ]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let filter = args.iter().find_map(|a| a.strip_prefix("--filter="));
    let only_machine = args.iter().find_map(|a| a.strip_prefix("--target="));
    std::fs::create_dir_all(WORKDIR).expect("workdir");
    let mut pass = 0;
    let mut fail = 0;
    let mut xfail = 0;
    for case in cases() {
        for machine in case.machines {
            if let Some(only) = only_machine {
                if machine.name() != only {
                    continue;
                }
            }
            let id = format!("{}_{}", case.name, machine.name());
            if let Some(f) = filter {
                if !id.contains(f) {
                    continue;
                }
            }
            let expected_fail = case.xfail.contains(machine);
            match run_case(&case, *machine, &id) {
                Ok(()) if expected_fail => {
                    println!("XPASS {id}: unexpectedly passed, update the expectation");
                    fail += 1;
                }
                Ok(()) => {
                    println!("PASS {id}");
                    pass += 1;
                }
                Err(_e) if expected_fail => {
                    println!("XFAIL {id}: tracked divergence");
                    xfail += 1;
                }
                Err(e) => {
                    println!("FAIL {id}: {e}");
                    fail += 1;
                }
            }
        }
    }
    println!("{pass} passed, {fail} failed, {xfail} xfailed");
    if fail > 0 {
        std::process::exit(1);
    }
}

fn run_case(case: &Case, machine: Machine, id: &str) -> Result<(), String> {
    let dir = PathBuf::from(WORKDIR);
    let obj_path = dir.join(format!("{id}.o"));
    // 1. C -> MIR -> Clif object through the real pipeline.
    build_module_object(case.module_src, machine.target(), machine.fe_target(), &obj_path)?;
    // 2. Harness + link.
    let harness_path = dir.join(format!("{id}_harness.c"));
    std::fs::write(&harness_path, case.harness_src).map_err(|e| e.to_string())?;
    let exe_path = dir.join(format!("{id}_t"));
    let mut link: Vec<String> = Vec::new();
    if machine == Machine::X86_64 {
        link.push("cc".to_string());
    } else {
        link.push("zig".to_string());
        link.push("cc".to_string());
        link.push("-target".to_string());
        link.push(machine.zig_triple().to_string());
    }
    link.push(harness_path.display().to_string());
    link.push(obj_path.display().to_string());
    if case.needs_glccrt && machine == Machine::X86_64 {
        link.push(glccrt_object(machine)?.display().to_string());
    }
    link.push("-o".to_string());
    link.push(exe_path.display().to_string());
    run(&link[0], &link[1..].iter().map(|s| s.as_str()).collect::<Vec<_>>())?;
    // 3. Run + compare.
    let output = if machine.runner().is_empty() {
        run_capture(exe_path.to_str().unwrap(), &[])?
    } else {
        let runner = machine.runner();
        let mut full = runner;
        full.push(exe_path.to_str().unwrap());
        run_capture(full[0], &full[1..])?
    };
    if output != case.expected_stdout {
        return Err(format!("stdout mismatch:\n  expected: {:?}\n  actual:   {:?}", case.expected_stdout, output));
    }
    Ok(())
}

fn build_module_object(src: &str, target: Target, fe_target: glcc_frontend::target::Target, obj_path: &Path) -> Result<(), String> {
    use cranelift_codegen::settings;
    use cranelift_codegen::settings::Configurable;

    // Frontend -> MIR.
    let mut interner = glcc_frontend::lexer::Interner::new();
    let mut sema = glcc_frontend::sema::Sema::with_target(fe_target);
    let tokens = glcc_frontend::lexer::Lexer::new(src, &mut interner).lex_all();
    let mut parser = glcc_frontend::parse::Parser::new(src, tokens, &mut sema, &mut interner);
    let tu = parser.parse_translation_unit();
    let mir_module = glcc_lower_mir::lower_translation_unit(&sema, &interner, &tu);
    let errors = glcc_mir::verify::verify_module(&mir_module);
    if !errors.is_empty() {
        return Err(format!("MIR verify errors: {errors:?}"));
    }
    // MIR -> Clif object.
    let mut flag_builder = settings::builder();
    flag_builder.set("opt_level", "none").map_err(|e| e.to_string())?;
    flag_builder.set("is_pic", "true").map_err(|e| e.to_string())?;
    flag_builder.set("preserve_frame_pointers", "true").map_err(|e| e.to_string())?;
    flag_builder.set("tls_model", "elf_gd").map_err(|e| e.to_string())?;
    let flags = settings::Flags::new(flag_builder);
    let isa: cranelift_codegen::isa::OwnedTargetIsa = match target.arch {
        Arch::X86_64 => cranelift_native::builder().map_err(|e| e.to_string())?.finish(flags).map_err(|e| e.to_string())?,
        _ => cranelift_codegen::isa::lookup(target.triple().parse().map_err(|_| "bad triple".to_string())?)
            .map_err(|e| e.to_string())?
            .finish(flags)
            .map_err(|e| e.to_string())?,
    };
    let builder = cranelift_object::ObjectBuilder::new(isa, "abi_probe", cranelift_module::default_libcall_names()).map_err(|e| e.to_string())?;
    let mut obj = cranelift_object::ObjectModule::new(builder);
    glcc_lower_cl::lower_module(&mir_module, &mut obj, target);
    let product = obj.finish();
    let bytes = product.emit().map_err(|e| e.to_string())?;
    std::fs::write(obj_path, bytes).map_err(|e| e.to_string())?;
    Ok(())
}

/// Compile the x86-64 variadic trampoline runtime once per run.
fn glccrt_object(machine: Machine) -> Result<PathBuf, String> {
    if machine != Machine::X86_64 {
        return Err("glccrt only needed on x86-64".to_string());
    }
    let out = PathBuf::from(WORKDIR).join("glccrt.o");
    if out.exists() {
        return Ok(out);
    }
    // Repo root relative to this file's crate is not stable; locate rt/
    // by walking up from the current executable is overkill: the probe
    // runs from the workspace, so a relative path works.
    let src = ["rt/glccrt.c", "../rt/glccrt.c", "../../rt/glccrt.c"]
        .iter()
        .map(PathBuf::from)
        .find(|p| p.exists())
        .ok_or("rt/glccrt.c not found (run from workspace root)".to_string())?;
    run(
        "cc",
        &[
            "-c",
            "-Os",
            "-fPIC",
            "-ftls-model=initial-exec",
            "-fno-asynchronous-unwind-tables",
            "-fno-stack-protector",
            src.to_str().unwrap(),
            "-o",
            out.to_str().unwrap(),
        ],
    )?;
    Ok(out)
}

fn run(prog: &str, args: &[&str]) -> Result<(), String> {
    let status = Command::new(prog).args(args).status().map_err(|e| format!("spawn {prog}: {e}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("{prog} exited {status}"))
    }
}

fn run_capture(prog: &str, args: &[&str]) -> Result<String, String> {
    let output = Command::new(prog).args(args).output().map_err(|e| format!("spawn {prog}: {e}"))?;
    if !output.status.success() {
        return Err(format!("{prog} exited {}", output.status));
    }
    String::from_utf8(output.stdout).map_err(|e| e.to_string())
}
