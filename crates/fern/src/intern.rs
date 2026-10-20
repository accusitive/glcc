//! String interning for session-level names.
//!
//! AST identifiers stay owned strings for now; the interner exists so
//! file paths and other long-lived session names have cheap handles, and
//! so later passes can move identifier storage here without changing the
//! session API.

use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Sym(pub u32);

impl Sym {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Debug, Default)]
pub struct Interner {
    map: HashMap<String, Sym>,
    strings: Vec<String>,
}

impl Interner {
    pub fn new() -> Interner {
        Interner::default()
    }

    pub fn intern(&mut self, text: &str) -> Sym {
        if let Some(sym) = self.map.get(text) {
            return *sym;
        }
        let sym = Sym(self.strings.len() as u32);
        self.strings.push(text.to_string());
        self.map.insert(text.to_string(), sym);
        sym
    }

    pub fn resolve(&self, sym: Sym) -> &str {
        &self.strings[sym.index()]
    }
}
