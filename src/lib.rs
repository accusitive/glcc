pub mod driver;
#[cfg(feature = "cranelift")]
pub mod winunwind;

#[cfg(not(any(feature = "cranelift", feature = "native")))]
compile_error!("glcc needs a code generator: enable the `cranelift` or `native` feature");
