//! x86-64 register table.

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RegClass {
    Gpr,
    Rip,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Reg {
    /// Encoding index into the ModRM reg/rm field (0-15).
    pub code: u8,
    /// Width in bytes (1, 2, 4, 8).
    pub size: u8,
    pub class: RegClass,
    /// ah/bh/ch/dh: byte registers 4-7 addressable without REX.
    pub h: bool,
}

impl Reg {
    pub const fn gpr(code: u8, size: u8) -> Reg {
        Reg { code, size, class: RegClass::Gpr, h: false }
    }

    pub const fn high(code: u8) -> Reg {
        Reg { code, size: 1, class: RegClass::Gpr, h: true }
    }

    pub const fn rip() -> Reg {
        Reg { code: 0, size: 0, class: RegClass::Rip, h: false }
    }

    pub fn is_gpr(&self) -> bool {
        self.class == RegClass::Gpr
    }

    pub fn is_64(&self) -> bool {
        self.is_gpr() && self.size == 8
    }

    /// Needs the REX prefix present just to name this register
    /// (spl/bpl/sil/dil, r8-r15 handled via code bits).
    pub fn needs_rex(&self) -> bool {
        self.is_gpr() && self.size == 1 && self.code >= 4 && !self.h
    }
}

pub const RIP: Reg = Reg::rip();

/// Case-insensitive lookup of the Intel register names.
pub fn lookup(name: &str) -> Option<Reg> {
    let n: String = name.to_ascii_lowercase();
    Some(match n.as_str() {
        "rip" => RIP,
        _ => return lookup_gpr(&n),
    })
}

fn lookup_gpr(n: &str) -> Option<Reg> {
    const NAMES64: [&str; 16] = [
        "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi", "r8", "r9",
        "r10", "r11", "r12", "r13", "r14", "r15",
    ];
    const NAMES32: [&str; 16] = [
        "eax", "ecx", "edx", "ebx", "esp", "ebp", "esi", "edi", "r8d", "r9d",
        "r10d", "r11d", "r12d", "r13d", "r14d", "r15d",
    ];
    const NAMES16: [&str; 16] = [
        "ax", "cx", "dx", "bx", "sp", "bp", "si", "di", "r8w", "r9w", "r10w",
        "r11w", "r12w", "r13w", "r14w", "r15w",
    ];
    const NAMES8: [&str; 12] = [
        "al", "cl", "dl", "bl", "spl", "bpl", "sil", "dil", "r8b", "r9b", "r10b", "r11b",
    ];
    const NAMES8_H: [&str; 4] = ["ah", "ch", "dh", "bh"];
    if let Some(code) = NAMES64.iter().position(|s| *s == n) {
        return Some(Reg::gpr(code as u8, 8));
    }
    if let Some(code) = NAMES32.iter().position(|s| *s == n) {
        return Some(Reg::gpr(code as u8, 4));
    }
    if let Some(code) = NAMES16.iter().position(|s| *s == n) {
        return Some(Reg::gpr(code as u8, 2));
    }
    if let Some(i) = NAMES8.iter().position(|s| *s == n) {
        // al/cl/dl/bl are codes 0-3; spl/bpl/sil/dil are the low bytes of
        // rsp/rbp/rsi/rdi and need REX.
        return Some(Reg::gpr(i as u8, 1));
    }
    if let Some(i) = NAMES8_H.iter().position(|s| *s == n) {
        return Some(Reg::high(i as u8 + 4));
    }
    // r12b..r15b
    if let Some(code) = n
        .strip_suffix('b')
        .and_then(|s| s.strip_prefix('r'))
        .and_then(|s| s.parse::<u8>().ok())
    {
        if (12..16).contains(&code) {
            return Some(Reg::gpr(code, 1));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes() {
        assert_eq!(lookup("RAX").unwrap().size, 8);
        assert_eq!(lookup("r15d").unwrap().size, 4);
        assert_eq!(lookup("r15d").unwrap().code, 15);
        assert_eq!(lookup("r12b").unwrap().code, 12);
        assert_eq!(lookup("sil").unwrap().code, 6);
        assert_eq!(lookup("bh").unwrap().code, 7);
        assert!(lookup("rbx").unwrap().is_64());
        assert!(lookup("rip").is_some());
        assert!(lookup("eaxx").is_none());
    }
}
