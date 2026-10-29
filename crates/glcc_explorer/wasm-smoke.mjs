// Smoke-test the wasm compiler module outside a browser:
//   node crates/glcc_explorer/wasm-smoke.mjs [module.wasm]
//
// Compiles a tiny program for every target, checks MIR + object bytes,
// then feeds it a syntax error and a warning and checks that both come
// back through the JSON diagnostics used by the editor.
import { readFile } from 'node:fs/promises';

const path = process.argv[2] ?? new URL('glcc-explorer.wasm', import.meta.url);
const bytes = await readFile(path);
const { instance } = await WebAssembly.instantiate(bytes, {});
let ex = instance.exports;
const enc = new TextEncoder();
const dec = new TextDecoder();

function compile(source, opt, target, flags = 0) {
  const src = enc.encode(source);
  const ptr = ex.glcc_alloc(src.length);
  new Uint8Array(ex.memory.buffer, ptr, src.length).set(src);
  try {
    const jsonLen = ex.glcc_compile(ptr, src.length, opt, target, flags);
    const json = JSON.parse(dec.decode(new Uint8Array(ex.memory.buffer, ex.glcc_result_ptr(), jsonLen)));
    const object = new Uint8Array(ex.memory.buffer, ex.glcc_object_ptr(), ex.glcc_object_len()).slice();
    return { json, object };
  } finally {
    ex.glcc_free(ptr, src.length);
  }
}

let failures = 0;
const check = (label, ok, detail) => {
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${label}${detail ? `: ${detail}` : ''}`);
  if (!ok) failures++;
};

for (const [target, name] of [[0, 'x86_64'], [1, 'aarch64'], [2, 'riscv64']]) {
  const { json, object } = compile('int add(int a, int b) { return a + b; }', 0, target);
  check(`${name} compile succeeds`, json.success, JSON.stringify(json.errors));
  check(`${name} emits MIR`, json.mir.includes('func @add'));
  check(`${name} emits object bytes`, object.length === json.object_len && object.length > 0, `${object.length} bytes`);
}

const { json: optJson } = compile('int add(int a, int b) { return a + b; }', 1, 0);
check('optimized compile succeeds', optJson.success, JSON.stringify(optJson.errors));
check('o1 optimized MIR differs from raw', optJson.mir !== optJson.mir_raw);

const { json: o0Json } = compile('int add(int a, int b) { return a + b; }', 0, 0);
check('o0 optimized MIR equals raw', o0Json.mir === o0Json.mir_raw, JSON.stringify(o0Json.errors));

const { json: commentJson } = compile('// leading\nint add(int a, int b) {\n  /* sum */ return a + b; // tail\n}', 0, 0);
check('comments compile', commentJson.success, JSON.stringify(commentJson.errors));

// `-fglcc-async` (flags bit 0): the async function is split into a
// start/resume pair on the way to the optimized MIR, while the raw MIR
// still carries the await. Without the flag the attribute is ignored.
const ASYNC_SRC = `_Future(int) add1(int x);
__attribute__((async)) int twice(int x) {
    int r = __builtin_await(add1(x));
    return r * 2;
}`;
const { json: asyncJson, object: asyncObject } = compile(ASYNC_SRC, 0, 0, 1);
check('async compile succeeds', asyncJson.success, JSON.stringify(asyncJson.errors));
check('async raw MIR keeps the await', asyncJson.mir_raw.includes('await'));
check('async MIR splits into $resume', asyncJson.mir.includes('$resume'));
check('async emits object bytes', asyncObject.length > 0, `${asyncObject.length} bytes`);

const { json: offJson } = compile('__attribute__((async)) int one(void) { return 1; }', 0, 0, 0);
check('async flag off ignores the attribute', offJson.success && !offJson.mir.includes('$resume'), JSON.stringify(offJson.errors));

// `-std=` rides bits 1..3 of flags (0 = default, 1 = c89, ... 5 = c23):
// C89 rejects `_Static_assert`, C23 accepts it.
const STATIC_ASSERT_SRC = '_Static_assert(1, "ok");\nint x;';
const { json: c89Json } = compile(STATIC_ASSERT_SRC, 0, 0, 1 << 1);
check('std=c89 rejects _Static_assert', !c89Json.success, JSON.stringify(c89Json.errors));
const { json: c23Json } = compile(STATIC_ASSERT_SRC, 0, 0, 5 << 1);
check('std=c23 accepts _Static_assert', c23Json.success, JSON.stringify(c23Json.errors));

// The standard also reaches `__STDC_VERSION__` through the bundled
// predefines (clang is not available inside wasm).
const versionSrc = '#if __STDC_VERSION__ >= 202311L\nint f(void) { return 1; }\n#else\nint f(void) { return 2; }\n#endif';
const { json: c23PpJson } = compile(versionSrc, 0, 0, 5 << 1);
check('std=c23 defines __STDC_VERSION__ 202311L', c23PpJson.success && c23PpJson.mir.includes('i32(1)'), JSON.stringify(c23PpJson.errors));
const { json: defaultPpJson } = compile(versionSrc, 0, 0, 0);
check('default keeps the C17 __STDC_VERSION__', defaultPpJson.success && defaultPpJson.mir.includes('i32(2)'), JSON.stringify(defaultPpJson.errors));

// Frontend errors are recoverable: they come back as JSON diagnostics
// (not a trap), and the same instance keeps compiling.
const { json: badJson } = compile('int x = ;', 0, 0);
check('syntax error fails without trapping', !badJson.success, JSON.stringify(badJson.errors));
const badDiagnostics = (badJson.diagnostics && badJson.diagnostics.diagnostics) || [];
check('syntax error carries JSON diagnostics', badDiagnostics.length > 0, JSON.stringify(badJson.diagnostics));
const firstError = badDiagnostics.find((d) => d.severity === 'error');
check('error diagnostic has a 1-based position', !!firstError && firstError.primary.line === 1 && firstError.primary.column > 0, JSON.stringify(firstError));

const { json: warningJson } = compile('int x = sizeof(void);', 0, 0);
const warnings = ((warningJson.diagnostics && warningJson.diagnostics.diagnostics) || []).filter((d) => d.severity === 'warning');
check('warnings share the diagnostics payload', warnings.length > 0, JSON.stringify(warningJson.diagnostics));

const { json: recoveredJson } = compile('int ok(void) { return 1; }', 0, 0);
check('instance keeps compiling after a recovered error', recoveredJson.success, JSON.stringify(recoveredJson.errors));

process.exit(failures ? 1 : 0);
