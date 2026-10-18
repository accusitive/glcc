// Smoke-test the wasm compiler module outside a browser:
//   node crates/glcc_explorer/wasm-smoke.mjs [module.wasm]
//
// Compiles a tiny program for every target, checks MIR + object bytes,
// then feeds it a syntax error (a panic inside wasm) and checks that
// the panic message comes back through the hook.
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

function panicMessage() {
  const len = ex.glcc_panic_len();
  return len ? dec.decode(new Uint8Array(ex.memory.buffer, ex.glcc_panic_ptr(), len)) : '';
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

// A frontend error panics inside wasm; the hook message must survive
// the trap, and a fresh instance must keep working.
try {
  compile('int x = ;', 0, 0);
  check('syntax error traps', false, 'no trap');
} catch (e) {
  const msg = panicMessage();
  check('syntax error traps', e instanceof WebAssembly.RuntimeError, String(e));
  check('panic message survives the trap', msg.length > 0, msg.split('\n')[0]);
  const fresh = await WebAssembly.instantiate(await WebAssembly.compile(bytes), {});
  ex = fresh.exports;
  const { json } = compile('int ok(void) { return 1; }', 0, 0);
  check('fresh instance compiles after a trap', json.success, JSON.stringify(json.errors));
}

process.exit(failures ? 1 : 0);
