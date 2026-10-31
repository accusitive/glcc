// mir_gen_be.cpp -- EDG IL -> glcc_mir text back end (v1 slice).
//
// Runs after the front end with the lowered IL resident in memory
// (DO_IL_LOWERING=1): the IL is C-shaped, so the walk mirrors
// c_gen_be.c's generate_C_output_file but emits MIR text per
// docs/edg-mir-backend.md instead of C. The same translation unit
// defines the back_end dispatcher: GLCPP_EDG_BACKEND=cgen routes to
// the legacy generated-C back end (compiled as c_gen_be_back_end),
// anything else takes this MIR path.
//
// Aggregate convention (mirrors crates/glcc_lower_cl/src/lib.rs): an
// aggregate-typed MIR value IS the address of its storage. Struct
// values flow as `struct<...>`-typed SSA values produced by loads and
// calls; places (addresses of storage) are `ptr(struct<...>)`. Array
// values decay at eok_array_to_pointer.
//
// Unsupported constructs (EH, bit-fields, reloc initializers, ...)
// print a diagnostic naming the construct and exit nonzero -- never
// silently wrong code.

#include <cinttypes>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "fe_common.h"
#include "cmd_line.h"
#include "il_file.h"
#include "il_read.h"
#include "il_write.h"

extern "C++" void c_gen_be_back_end(void);

namespace {

[[noreturn]] void unsupported(const char *what) {
  std::fprintf(stderr, "glcc mir_gen_be: unsupported construct: %s\n", what);
  std::exit(3);
}

[[noreturn]] void unsupported_f(const char *fmt, ...) {
  std::va_list ap;
  va_start(ap, fmt);
  std::fprintf(stderr, "glcc mir_gen_be: unsupported construct: ");
  std::vfprintf(stderr, fmt, ap);
  va_end(ap);
  std::fputc('\n', stderr);
  std::exit(3);
}

bool type_is_aggregate(a_type_ptr type) {
  a_type_kind kind = type->kind;
  return kind == (a_type_kind)tk_struct || kind == (a_type_kind)tk_class ||
         kind == (a_type_kind)tk_union || kind == (a_type_kind)tk_array;
}

bool type_is_bool(a_type_ptr type) {
  return type->kind == (a_type_kind)tk_integer &&
         type->variant.integer.bool_type;
}

bool int_kind_is_signed(an_integer_kind kind) {
  switch (kind) {
    case ik_signed_char:
    case ik_short:
    case ik_int:
    case ik_long:
    case ik_long_long:
    case ik_int128:
      return TRUE;
    default:
      return FALSE;
  }
}

unsigned int_prim_bits(an_integer_kind kind) {
  switch (kind) {
    case ik_char:
    case ik_signed_char:
    case ik_unsigned_char:
      return 8;
    case ik_short:
    case ik_unsigned_short:
      return 16;
    case ik_int:
    case ik_unsigned_int:
      return 32;
    case ik_long:
    case ik_unsigned_long:
    case ik_long_long:
    case ik_unsigned_long_long:
      return 64;
    case ik_int128:
    case ik_unsigned_int128:
      return 128;
    default:
      unsupported_f("integer kind %d", (int)kind);
  }
}

unsigned int_bits(a_type_ptr type) {
  type = skip_typerefs(type);
  if (type->kind == (a_type_kind)tk_integer) {
    if (type->variant.integer.bool_type) return 8;
    return int_prim_bits(type->variant.integer.int_kind);
  }
  unsupported_f("int bits of non-integer type kind %d", (int)type->kind);
}

bool il_int_signed(a_type_ptr type) {
  type = skip_typerefs(type);
  if (type->kind != (a_type_kind)tk_integer) {
    unsupported_f("signedness of non-integer type kind %d", (int)type->kind);
  }
  return int_kind_is_signed(type->variant.integer.int_kind);
}

const char *float_kind_name(a_float_kind kind) {
  switch (kind) {
    case fk_float:
    case fk_std_float32:
      return "f32";
    case fk_double:
    case fk_std_float64:
      return "f64";
    case fk_long_double:
    case fk_float80:
      return "f80";
    case fk_float128:
    case fk_std_float128:
      return "f128";
    default:
      unsupported_f("float kind %d", (int)kind);
  }
}

int float_kind_bytes(a_float_kind kind) {
  switch (kind) {
    case fk_float:
    case fk_std_float32:
      return 4;
    case fk_double:
    case fk_std_float64:
      return 8;
    case fk_long_double:
    case fk_float80:
    case fk_float128:
    case fk_std_float128:
      return 16;
    default:
      return 0;
  }
}

bool pointee_is_struct(a_type_ptr pointee_il) {
  a_type_ptr pointee = skip_typerefs(pointee_il);
  a_type_kind kind = pointee->kind;
  return kind == (a_type_kind)tk_struct || kind == (a_type_kind)tk_class ||
         kind == (a_type_kind)tk_union;
}

// Whether a base expression of the given IL type navigates through an
// erased struct pointer (needing byte-offset arithmetic instead of a
// gep step).
bool base_pointee_erased(a_type_ptr base_il) {
  a_type_ptr ty = skip_typerefs(base_il);
  return ty->kind == (a_type_kind)tk_pointer &&
         pointee_is_struct(ty->variant.pointer.type);
}

// MIR type text for an IL type. Typerefs are transparent and types are
// inline, so nothing needs declaring ahead of use.
std::string ty_name(a_type_ptr type) {
  type = skip_typerefs(type);
  switch (type->kind) {
    case tk_void:
      return "void";
    case tk_integer:
      if (type->variant.integer.bool_type) return "bool";
      return "i" + std::to_string(int_prim_bits(type->variant.integer.int_kind));
    case tk_float:
      return float_kind_name(type->variant.float_kind);
    case tk_pointer:
      if (type->variant.pointer.is_reference) {
        unsupported("C++ reference type");
      }
      // Struct pointees erase to void (mirrors glcc_lower_mir's
      // lower_ty): navigating into one re-derives the layout from the
      // IL per hop, and keeping the definition would recurse forever
      // on self-referential structs (struct S { S *next; }, FILE).
      if (pointee_is_struct(type->variant.pointer.type)) {
        return "ptr(void)";
      }
      return "ptr(" + ty_name(type->variant.pointer.type) + ")";
    case tk_array: {
      if (type->variant.array.is_vla ||
          type->variant.array.is_variable_size_array) {
        unsupported("variable-length array");
      }
      unsigned long long n = (unsigned long long)
          type->variant.array.variant.number_of_elements;
      return "[" + std::to_string(n) + " x " +
             ty_name(type->variant.array.element_type) + "]";
    }
    case tk_struct:
    case tk_class:
    case tk_union: {
      std::string s = "struct<" +
                      std::to_string((unsigned long long)type->size) + ">{";
      for (a_field_ptr field = type->variant.class_struct_union.field_list;
           field != NULL; field = field->next) {
        if (field->is_bit_field) unsupported("bit-fields");
        s += std::to_string((unsigned long long)field->offset);
        s += ":";
        s += ty_name(field->type);
        s += ",";
      }
      s += "}";
      return s;
    }
    case tk_routine:
      // Function types have no MIR type; function pointers are plain
      // addresses (signatures travel on call.ind and the callee).
      return "ptr(void)";
    default:
      unsupported_f("type kind %d", (int)type->kind);
  }
}

std::string entity_name(a_source_correspondence *corresp, const char *kind_desc) {
  if (corresp->name != NULL && corresp->name[0] != '\0') {
    return std::string(corresp->name);
  }
  unsupported_f("%s without a name", kind_desc);
}

// Stack-synthesized scalar types, for values the IL does not carry
// (e.g. an i64 zero for pointer arithmetic). Nothing retains pointers
// to them and ty_name only reads the fields set here.
a_type synth_int_type(unsigned bits, bool is_signed) {
  a_type t;
  std::memset(&t, 0, sizeof(t));
  t.kind = (a_type_kind)tk_integer;
  an_integer_kind kind;
  if (bits <= 8) {
    kind = is_signed ? ik_signed_char : ik_unsigned_char;
  } else if (bits <= 16) {
    kind = is_signed ? ik_short : ik_unsigned_short;
  } else if (bits <= 32) {
    kind = is_signed ? ik_int : ik_unsigned_int;
  } else {
    kind = is_signed ? ik_long_long : ik_unsigned_long_long;
  }
  t.variant.integer.int_kind = kind;
  return t;
}

a_type synth_bool_type() {
  a_type t;
  std::memset(&t, 0, sizeof(t));
  t.kind = (a_type_kind)tk_integer;
  t.variant.integer.bool_type = TRUE;
  t.variant.integer.int_kind = ik_unsigned_char;
  return t;
}

a_type synth_float_type(const char *mir_name) {
  a_type t;
  std::memset(&t, 0, sizeof(t));
  t.kind = (a_type_kind)tk_float;
  a_float_kind kind = fk_double;
  if (std::strcmp(mir_name, "f32") == 0) kind = fk_float;
  if (std::strcmp(mir_name, "f128") == 0) kind = fk_float128;
  t.variant.float_kind = kind;
  return t;
}

// ---------------------------------------------------------------------------
// Static initializer text
// ---------------------------------------------------------------------------

std::string const_init_text(a_constant_ptr con);

std::string int_const_text(a_constant_ptr con) {
  unsigned bits = int_bits(con->type);
  if (bits > 64) unsupported("integer constant wider than 64 bits");
  a_host_large_integer value;
  a_boolean err;
  conv_integer_value_to_host_large_integer(&con->variant.integer_value,
                                           il_int_signed(con->type),
                                           &value, &err);
  if (err) unsupported("integer constant conversion failed");
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%lld", (long long)value);
  return buf;
}

std::string float_bits_text(a_float_kind kind, unsigned __int128 bits) {
  int nbytes = float_kind_bytes(kind);
  char buf[48];
  std::snprintf(buf, sizeof(buf), "%s:0x%016" PRIx64 "%016" PRIx64,
                float_kind_name(kind), (unsigned long long)(bits >> 64),
                (unsigned long long)bits);
  std::string s(buf);
  if (nbytes <= 8) {
    // Drop the zero high half so 8-byte constants read like the spec
    // example: f64:0x3ff0000000000000.
    std::size_t start = s.find("0x") + 2;
    std::size_t end = start;
    while (end + 1 < s.size() && s[end] == '0' && s[end + 1] == '0') end += 2;
    if (s[end] == '0' && (s.size() - end) % 2 == 0) end++;
    s = s.substr(0, start - 2) + "0x" + s.substr(end);
  }
  return s;
}

std::string float_const_text(a_constant_ptr con) {
  a_type_ptr type = skip_typerefs(con->type);
  a_float_kind kind = type->variant.float_kind;
  int nbytes = float_kind_bytes(kind);
  // The internal float value holds the host representation, which on
  // x86-64 is the IEEE bit pattern for f32/f64 and the 16-byte x87
  // extended carrier for f80 -- exactly the MIR carrier.
  unsigned __int128 bits = 0;
  std::memcpy(&bits, con->variant.float_value.bytes, (size_t)nbytes);
  return float_bits_text(kind, bits);
}

std::string string_const_text(a_constant_ptr con) {
  static const char hex[] = "0123456789abcdef";
  std::string s = "bytes<";
  for (a_targ_size_t i = 0; i < con->variant.string.length; i++) {
    unsigned char byte = (unsigned char)con->variant.string.value[i];
    s.push_back(hex[byte >> 4]);
    s.push_back(hex[byte & 0xf]);
  }
  s += ">";
  return s;
}

// Dense aggregate init: one element per array element / struct field,
// `zeroinit` for elements the constant does not cover.
std::string aggregate_const_text(a_constant_ptr con) {
  a_type_ptr type = skip_typerefs(con->type);
  std::vector<a_constant_ptr> elems;
  for (a_constant_ptr elem = con->variant.aggregate.first_constant; elem != NULL;
       elem = elem->next) {
    if (elem->kind == (a_constant_repr_kind)ck_designator ||
        elem->kind == (a_constant_repr_kind)ck_init_repeat) {
      unsupported("designated/repeat aggregate initializers");
    }
    elems.push_back(elem);
  }
  unsigned long long count = 0;
  if (type->kind == (a_type_kind)tk_array) {
    count = (unsigned long long)type->variant.array.variant.number_of_elements;
  } else {
    for (a_field_ptr f = type->variant.class_struct_union.field_list; f != NULL;
         f = f->next) {
      count++;
    }
  }
  std::string s = "[";
  for (unsigned long long i = 0; i < count; i++) {
    if (i > 0) s += ", ";
    s += i < elems.size() ? const_init_text(elems[i]) : "zeroinit";
  }
  s += "]";
  return s;
}

std::string string_global_name(a_constant_ptr con);

std::string address_const_text(a_constant_ptr con) {
  // Relocation initializer: `reloc @sym` optionally followed by a byte
  // offset, `+<n>`, with no intervening whitespace (see
  // docs/edg-mir-backend.md "Format amendments").
  std::string sym;
  if (con->variant.address.kind == (an_address_base_kind)abk_variable) {
    a_variable_ptr var = con->variant.address.variant.variable;
    sym = "@" + entity_name(&var->source_corresp, "variable");
  } else if (con->variant.address.kind == (an_address_base_kind)abk_routine) {
    sym = "@" + entity_name(&con->variant.address.variant.routine->source_corresp,
                            "routine");
  } else if (con->variant.address.kind == (an_address_base_kind)abk_constant) {
    a_constant_ptr inner = con->variant.address.variant.constant;
    if (inner->kind != (a_constant_repr_kind)ck_string) {
      unsupported("address-of-nonstring constant initializer");
    }
    sym = "@" + string_global_name(inner);
  } else {
    unsupported_f("address constant kind %d in static initializer",
                  (int)con->variant.address.kind);
  }
  if (con->variant.address.offset != 0) {
    sym += "+" + std::to_string((unsigned long long)con->variant.address.offset);
  }
  return "reloc " + sym;
}

std::string const_init_text(a_constant_ptr con) {
  switch (con->kind) {
    case ck_integer:
      return int_const_text(con);
    case ck_float:
      return float_const_text(con);
    case ck_string:
      return string_const_text(con);
    case ck_aggregate:
      return aggregate_const_text(con);
    case ck_address:
      return address_const_text(con);
    default:
      unsupported_f("constant kind %d in static initializer", (int)con->kind);
  }
}

} // namespace

BEGIN_EDG_NAMESPACE

namespace {

FILE *f_out = NULL;
int anon_counter = 0;
// String literal constants referenced from function bodies, mapped to
// synthesized global names. Collected in a pre-pass so the `.global`
// lines precede the bodies that reference them.
std::map<a_constant_ptr, std::string> string_globals;
std::vector<std::string> pending_string_lines;
// Module items discovered while emitting function bodies (unnamed
// static temporaries), flushed at end of file.
std::vector<std::string> pending_module_lines;
// Global names already declared (.global/.dataextern), so lazily
// registered temporaries do not collide with the module walk.
std::map<std::string, bool> declared_globals;
// Names for unnamed static-storage variables, per IL entry: every
// reference to the same variable must name the same object.
std::map<a_variable_ptr, std::string> var_global_names;

std::string next_anon_name(const char *prefix) {
  return std::string(prefix) + std::to_string(++anon_counter);
}

// Synthesized IL type for a string-literal constant: `char [len]`.
// The constant's own type entry may not survive the IL round trip
// (only "needed" types are written), but length and character kind
// do. The returned pointer is stable for the process lifetime.
a_type_ptr string_array_il_type(a_constant_ptr con) {
  if (con->character_kind != (a_character_kind)chk_char) {
    unsupported("non-char string literal");
  }
  static std::map<unsigned long long, a_type *> cache;
  unsigned long long len = (unsigned long long)con->variant.string.length;
  auto it = cache.find(len);
  if (it != cache.end()) return it->second;
  a_type *pair = (a_type *)std::malloc(sizeof(a_type) * 2);
  std::memset(pair, 0, sizeof(a_type) * 2);
  pair[0].kind = (a_type_kind)tk_integer;
  pair[0].variant.integer.int_kind = ik_char;
  pair[1].kind = (a_type_kind)tk_array;
  pair[1].variant.array.element_type = &pair[0];
  pair[1].variant.array.variant.number_of_elements =
      (a_targ_size_t)len;
  cache.emplace(len, &pair[1]);
  return &pair[1];
}

// The (registered) global name for a string-literal constant; also
// schedules its .global line. Shared by function-body slots and
// static-initializer relocations.
std::string string_global_name(a_constant_ptr con) {
  auto it = string_globals.find(con);
  if (it != string_globals.end()) return it->second;
  std::string name = next_anon_name(".str.");
  string_globals.emplace(con, name);
  pending_string_lines.push_back(".global @" + name + " const ty=" +
                                 ty_name(string_array_il_type(con)) +
                                 " init=" + string_const_text(con));
  return name;
}

// ---------------------------------------------------------------------------
// Function body emitter
// ---------------------------------------------------------------------------

a_type_ptr static_float_il_type(a_float_kind kind) {
  // One process-lifetime type per MIR float kind.
  struct Entry {
    a_float_kind kind;
    a_type ty;
  };
  static Entry entries[4] = {};
  static unsigned count = 0;
  for (unsigned i = 0; i < count; i++) {
    if (entries[i].kind == kind) return &entries[i].ty;
  }
  a_type t;
  std::memset(&t, 0, sizeof(t));
  t.kind = (a_type_kind)tk_float;
  t.variant.float_kind = kind;
  entries[count].kind = kind;
  entries[count].ty = t;
  return &entries[count++].ty;
}

// Process-lifetime synthesized pointer type for an IL pointee: address
// values carry ptr(T) so later ptr->int arithmetic types correctly.
// Keyed by the IL pointee entry, which lives in the (permanent) IL
// regions.
a_type_ptr synth_ptr_il_type(a_type_ptr pointee) {
  static std::map<a_type_ptr, a_type *> cache;
  auto it = cache.find(pointee);
  if (it != cache.end()) return it->second;
  a_type *t = (a_type *)std::malloc(sizeof(a_type));
  std::memset(t, 0, sizeof(a_type));
  t->kind = (a_type_kind)tk_pointer;
  t->variant.pointer.type = pointee;
  cache.emplace(pointee, t);
  return t;
}

struct Val {
  int n = -1; // %n; unused when is_const.
  a_type_ptr ty = NULL;
  // Constants ride inline as operands (the MIR text format has no
  // constant instructions); they carry their payload until printed.
  enum ConstKind { NotConst, IntC, FloatC, MiscC } const_kind = NotConst;
  long long int_val = 0;
  unsigned __int128 float_bits = 0;
  const char *misc = NULL; // "undef" / "zeroinit" / "null"
};

struct FuncEmitter {
  int next_value = 0;
  int next_block = 1; // b0 is the entry block.
  int cur_block = 0;
  bool terminated = false;
  std::vector<a_variable_ptr> params;
  std::map<a_variable_ptr, int> slots;         // variable -> alloca value id
  std::map<a_label_ptr, int> label_blocks;     // label -> block id
  // Block parameters, declared on the label line: (block, pos) ->
  // (value id, type text).
  std::map<std::pair<int, int>, std::pair<int, std::string>> block_params;
  std::map<a_dynamic_init_ptr, bool> inits_done_at_decl;
  std::map<a_statement_ptr, int> case_blocks; // stmk_switch_case -> block
  a_scope_ptr func_scope = NULL;
  a_routine_ptr func_rout = NULL;

  // ---- plumbing -------------------------------------------------------

  int new_value() { return next_value++; }

  int new_block() { return next_block++; }

  // A block parameter is allocated once; every br targeting the block
  // at that position passes the same %n, and the label line declares
  // it with its type.
  int block_param(int block, int pos, a_type_ptr ty) {
    auto key = std::make_pair(block, pos);
    auto it = block_params.find(key);
    if (it != block_params.end()) return it->second.first;
    int v = new_value();
    block_params.emplace(key, std::make_pair(v, ty_name(ty)));
    return v;
  }

  void start_block(int block) {
    if (!terminated && block != cur_block) {
      std::fprintf(f_out, "  br b%d\n", block);
    }
    bool have_params = false;
    for (auto &entry : block_params) {
      if (entry.first.first == block) have_params = true;
    }
    if (have_params) {
      std::fprintf(f_out, "b%d(", block);
      bool first = true;
      for (auto &entry : block_params) {
        if (entry.first.first != block) continue;
        if (!first) std::fprintf(f_out, ", ");
        first = false;
        std::fprintf(f_out, "%%%d:%s", entry.second.first,
                     entry.second.second.c_str());
      }
      std::fprintf(f_out, "):\n");
    } else {
      std::fprintf(f_out, "b%d:\n", block);
    }
    cur_block = block;
    terminated = false;
  }

  void emit(const char *fmt, ...) {
    std::va_list ap;
    va_start(ap, fmt);
    std::fputs("  ", f_out);
    std::vfprintf(f_out, fmt, ap);
    va_end(ap);
    std::fputc('\n', f_out);
  }

  // Unreachable code after a terminator still needs somewhere to go.
  void ensure_open() {
    if (terminated) start_block(new_block());
  }

  void emit_br(int target) {
    ensure_open();
    emit("br b%d", target);
    terminated = true;
  }

  void emit_br_with(int target, int pos, int value) {
    ensure_open();
    emit("br b%d(%%%d)", target, value);
    (void)pos;
    terminated = true;
  }

  void emit_brif(Val cond, int then_block, int then_arg, int else_block,
                 int else_arg) {
    ensure_open();
    if (cond.const_kind == Val::IntC) {
      // A constant condition never reaches the text as a brif (the
      // parser requires a value operand): branch directly.
      emit_br(cond.int_val != 0 ? then_block : else_block);
      return;
    }
    (void)then_arg;
    (void)else_arg;
    emit("brif %%%d b%d b%d", cond.n, then_block, else_block);
    terminated = true;
  }

  // ---- values ----------------------------------------------------------

  Val val(int n, a_type_ptr ty) {
    Val v;
    v.n = n;
    v.ty = ty;
    return v;
  }

  Val const_int_val(long long value, a_type_ptr ty) {
    Val v;
    v.ty = ty;
    v.const_kind = Val::IntC;
    v.int_val = value;
    return v;
  }

  Val const_misc_val(a_type_ptr ty, const char *text) {
    Val v;
    v.ty = ty;
    v.const_kind = Val::MiscC;
    v.misc = text;
    return v;
  }

  // Operand spelling: a %n reference or an inline constant.
  std::string opnd(const Val &v) {
    if (v.const_kind == Val::NotConst) return "%" + std::to_string(v.n);
    if (v.const_kind == Val::IntC) return std::to_string(v.int_val);
    if (v.const_kind == Val::MiscC) return v.misc;
    return float_bits_text(skip_typerefs(v.ty)->variant.float_kind,
                           v.float_bits);
  }

  Val inst1(const char *op, a_type_ptr ty) {
    ensure_open();
    int n = new_value();
    emit("%%%d:%s = %s", n, ty_name(ty).c_str(), op);
    return val(n, ty);
  }

  Val int_const(long long value, a_type_ptr ty) {
    return const_int_val(value, ty);
  }

  Val zero_const(a_type_ptr ty) {
    ty = skip_typerefs(ty);
    if (ty->kind == (a_type_kind)tk_float) {
      Val v;
      v.ty = ty;
      v.const_kind = Val::FloatC;
      return v; // all-zero bits = +0.0
    }
    return int_const(0, ty);
  }

  Val undef_val(a_type_ptr ty) { return const_misc_val(ty, "undef"); }

  Val float_const_bits(a_float_kind kind, unsigned __int128 bits) {
    Val v;
    v.ty = static_float_il_type(kind);
    v.const_kind = Val::FloatC;
    v.float_bits = bits;
    return v;
  }

  long long int_const_value(a_constant_ptr con) {
    unsigned bits = int_bits(con->type);
    if (bits > 64) unsupported("integer constant wider than 64 bits");
    a_host_large_integer value;
    a_boolean err;
    conv_integer_value_to_host_large_integer(&con->variant.integer_value,
                                             il_int_signed(con->type),
                                             &value, &err);
    if (err) unsupported("integer constant conversion failed");
    return (long long)value;
  }

  // ---- memory ----------------------------------------------------------

  Val alloca(a_type_ptr type) {
    ensure_open();
    int n = new_value();
    emit("%%%d:ptr(%s) = alloca %s", n, ty_name(type).c_str(),
         ty_name(type).c_str());
    return val(n, synth_ptr_il_type(type));
  }

  Val load(Val addr, a_type_ptr type) {
    ensure_open();
    int n = new_value();
    emit("%%%d:%s = load %%%d", n, ty_name(type).c_str(), addr.n);
    return val(n, type);
  }

  void store(Val addr, Val value) {
    ensure_open();
    emit("store %%%d, %s", addr.n, opnd(value).c_str());
  }

  // Stores a value into storage; aggregates copy with memcpy (their
  // value is the address of the storage to copy from).
  void store_or_copy(Val addr, Val value, a_type_ptr type) {
    if (type_is_aggregate(skip_typerefs(type))) {
      memcpy_bytes(addr, value, (unsigned long long)skip_typerefs(type)->size);
    } else {
      store(addr, value);
    }
  }

  void memcpy_bytes(Val dst, Val src, unsigned long long len) {
    ensure_open();
    emit("memcpy %%%d, %%%d, len=%llu", dst.n, src.n, len);
  }

  Val gep(Val base, a_type_ptr elem_ty, const std::string &steps) {
    ensure_open();
    int n = new_value();
    emit("%%%d:ptr(%s) = gep %%%d %s", n, ty_name(elem_ty).c_str(), base.n,
         steps.c_str());
    return val(n, synth_ptr_il_type(elem_ty));
  }

  // GEP step text for an index: a constant index spells a bare
  // ConstIndex (the `index %v` form requires a value operand);
  // non-constant indices use `index %n`.
  std::string gep_step(Val idx, bool *is_const) {
    if (idx.const_kind == Val::IntC && idx.int_val >= 0) {
      *is_const = true;
      return std::to_string(idx.int_val);
    }
    *is_const = false;
    return "index %" + std::to_string(idx.n);
  }

  // base + offset bytes, retyped to ptr(result_pointee) -- used to
  // navigate erased struct pointers (whose MIR pointee is void).
  Val byte_offset_addr(Val base, unsigned long long offset,
                       a_type_ptr result_pointee) {
    a_type i64 = synth_int_type(64, TRUE);
    Val as_int = convert(base, base.ty, &i64);
    if (offset != 0) {
      Val off = int_const((long long)offset, &i64);
      ensure_open();
      int n = new_value();
      emit("%%%d:i64 = add %s, %s", n, opnd(as_int).c_str(),
           opnd(off).c_str());
      as_int = val(n, &i64);
    }
    ensure_open();
    int m = new_value();
    emit("%%%d:ptr(%s) = inttoptr %s", m, ty_name(result_pointee).c_str(),
         opnd(as_int).c_str());
    return val(m, synth_ptr_il_type(result_pointee));
  }

  // base + idx * scale bytes (scaled indexing through an erased
  // struct pointer or struct-element array pointer).
  Val scaled_byte_addr(Val base, Val idx, unsigned long long scale,
                       a_type_ptr result_pointee) {
    a_type i64 = synth_int_type(64, TRUE);
    Val as_int = convert(base, base.ty, &i64);
    Val wide = convert(idx, idx.ty, &i64);
    if (scale != 1) {
      Val n_scale = int_const((long long)scale, &i64);
      ensure_open();
      int n = new_value();
      emit("%%%d:i64 = mul %s, %s", n, opnd(wide).c_str(),
           opnd(n_scale).c_str());
      wide = val(n, &i64);
    }
    ensure_open();
    int n = new_value();
    emit("%%%d:i64 = add %s, %s", n, opnd(as_int).c_str(), opnd(wide).c_str());
    Val sum = val(n, &i64);
    ensure_open();
    int m = new_value();
    emit("%%%d:ptr(%s) = inttoptr %s", m, ty_name(result_pointee).c_str(),
         opnd(sum).c_str());
    return val(m, synth_ptr_il_type(result_pointee));
  }

  // ---- conversions -------------------------------------------------------

  Val convert(Val v, a_type_ptr from_il, a_type_ptr to_il);
  Val convert_const(Val v, a_type_ptr from, a_type_ptr to);
  Val double_const_to(a_type_ptr to, double d);

  Val to_bool(Val v) {
    a_type_ptr ty = skip_typerefs(v.ty);
    a_type bty = synth_bool_type();
    if (v.const_kind != Val::NotConst) {
      long long truth;
      if (v.const_kind == Val::IntC) {
        truth = v.int_val != 0;
      } else if (v.const_kind == Val::MiscC) {
        truth = 0; // null
      } else {
        truth = v.float_bits != 0;
      }
      return const_int_val(truth, &bty);
    }
    if (type_is_bool(ty)) return v;
    if (ty->kind == (a_type_kind)tk_float) {
      Val zero = zero_const(ty);
      int n = new_value();
      ensure_open();
      emit("%%%d:bool = fcmp.ne %s, %s", n, opnd(v).c_str(),
           opnd(zero).c_str());
      return val(n, &bty);
    }
    if (ty->kind == (a_type_kind)tk_pointer) {
      a_type i64 = synth_int_type(64, TRUE);
      Val as_int = convert(v, ty, &i64);
      Val zero = int_const(0, &i64);
      int n = new_value();
      ensure_open();
      emit("%%%d:bool = icmp.ne %s, %s", n, opnd(as_int).c_str(),
           opnd(zero).c_str());
      return val(n, &bty);
    }
    Val zero = zero_const(ty);
    int n = new_value();
    ensure_open();
    emit("%%%d:bool = icmp.ne %s, %s", n, opnd(v).c_str(),
         opnd(zero).c_str());
    return val(n, &bty);
  }

  Val icmp(const char *pred, Val a, Val b, a_type_ptr result_ty) {
    a_type bty = synth_bool_type();
    ensure_open();
    int n = new_value();
    emit("%%%d:bool = icmp.%s %s, %s", n, pred, opnd(a).c_str(),
         opnd(b).c_str());
    Val res = val(n, &bty);
    if (type_is_bool(skip_typerefs(result_ty))) return res;
    return convert(res, &bty, result_ty);
  }

  Val fcmp(const char *pred, Val a, Val b, a_type_ptr result_ty) {
    a_type bty = synth_bool_type();
    ensure_open();
    int n = new_value();
    emit("%%%d:bool = fcmp.%s %s, %s", n, pred, opnd(a).c_str(),
         opnd(b).c_str());
    Val res = val(n, &bty);
    if (type_is_bool(skip_typerefs(result_ty))) return res;
    return convert(res, &bty, result_ty);
  }

  // ---- variable storage --------------------------------------------------

  Val var_storage(a_variable_ptr var) {
    auto it = slots.find(var);
    if (it != slots.end()) return val(it->second, var->type);
    if (var->is_parameter) {
      unsupported_f("parameter '%s' referenced without a slot",
                    var->source_corresp.name ? var->source_corresp.name : "?");
    }
    a_storage_class sc = var->storage_class;
    if (sc == (a_storage_class)sc_auto || sc == (a_storage_class)sc_register) {
      Val slot = alloca(var->type);
      slots.emplace(var, slot.n);
      return slot;
    }
    // Static storage duration: a global. Unnamed compiler-generated
    // static temporaries (guard objects, troublesome-aggregate
    // constant holders) get a synthesized name and a deferred
    // .global line.
    std::string name;
    if (var->source_corresp.name != NULL &&
        var->source_corresp.name[0] != '\0') {
      name = var->source_corresp.name;
    } else {
      auto it = var_global_names.find(var);
      if (it != var_global_names.end()) {
        name = it->second;
      } else {
        name = next_anon_name(".temp.");
        var_global_names.emplace(var, name);
        if (!declared_globals.count(name)) {
          declared_globals.emplace(name, true);
          pending_module_lines.push_back(".global @" + name + " ty=" +
                                         ty_name(var->type) +
                                         " init=zeroinit");
        }
      }
    }
    ensure_open();
    int n = new_value();
    emit("%%%d:ptr(%s) = globaladdr @%s", n, ty_name(var->type).c_str(),
         name.c_str());
    return val(n, synth_ptr_il_type(var->type));
  }

  Val string_slot(a_constant_ptr con) {
    std::string name = string_global_name(con);
    a_type_ptr arr_ty = string_array_il_type(con);
    ensure_open();
    int n = new_value();
    emit("%%%d:ptr(%s) = globaladdr @%s", n, ty_name(arr_ty).c_str(),
         name.c_str());
    return val(n, synth_ptr_il_type(arr_ty));
  }

  // ---- expressions ---------------------------------------------------------

  // Evaluates an expression. Scalar prvalues produce the value; a
  // glvalue (is_lvalue/is_xvalue) produces the ADDRESS of the object
  // (ptr(T)); an aggregate prvalue produces the aggregate-typed value,
  // which itself is an address of a copy (the MIR convention).
  Val eval(an_expr_node_ptr expr);

  // Address of the storage backing an lvalue expression.
  Val place(an_expr_node_ptr expr);

  // The two-node operand of a field selection, as a base address. An
  // aggregate rvalue operand is copied to a temporary first so its
  // members have stable storage.
  Val field_base(an_expr_node_ptr obj);

  Val binop(an_expr_operator_kind op, Val a, Val b, a_type_ptr result_ty,
            a_type_ptr operand_ty);

  Val call_emit(an_expr_node_ptr selector, an_expr_node_ptr args,
                a_type_ptr result_ty);

  Val call_addr(a_routine_ptr rout);
  Val increment(Val v, bool up);
  Val pointer_arith(an_expr_operator_kind op, Val base, Val idx,
                    a_type_ptr result_ty);

  // ---- statements ----------------------------------------------------------

  void walk_statement(a_statement_ptr stmt);
  void walk_statements(a_statement_ptr stmt);
  void declare_block_locals(a_statement_ptr block);
  void do_dynamic_init(a_dynamic_init_ptr dip);
  void store_const_into(Val addr, a_constant_ptr con, a_type_ptr type);
  int label_block(a_label_ptr label);
};

// Compile-time conversion of a constant operand: the MIR text format
// has no constant instructions, so casts of constants must fold.
Val FuncEmitter::convert_const(Val v, a_type_ptr from, a_type_ptr to) {
  if (v.const_kind == Val::MiscC) {
    // null stays null across pointer casts; anything else to an
    // integer is out of reach for v1.
    if (to->kind == (a_type_kind)tk_pointer) {
      v.ty = to;
      return v;
    }
    unsupported("constant pointer used as an integer");
  }
  if (v.const_kind == Val::IntC) {
    if (to->kind == (a_type_kind)tk_integer) {
      // Bit pattern in the low bits of int_val is already correct for
      // truncation; zero/sign extension re-derives the mathematical
      // value of the result width.
      unsigned fb = int_bits(from);
      unsigned tb = int_bits(to);
      long long result = v.int_val;
      if (tb > fb) {
        if (!il_int_signed(from)) {
          result = (long long)(unsigned long long)v.int_val &
                   (fb >= 64 ? -1 : (long long)((1ULL << fb) - 1));
        }
      }
      if (type_is_bool(to)) {
        result = result != 0;
      }
      v.int_val = result;
      v.ty = to;
      return v;
    }
    if (to->kind == (a_type_kind)tk_float) {
      double d = il_int_signed(from)
                     ? (double)v.int_val
                     : (double)(unsigned long long)v.int_val;
      return double_const_to(to, d);
    }
    unsupported_f("constant conversion %s -> %s", ty_name(from).c_str(),
                  ty_name(to).c_str());
  }
  // Float constant.
  if (to->kind == (a_type_kind)tk_float) {
    unsigned fb = float_kind_bytes(from->variant.float_kind);
    unsigned tb = float_kind_bytes(to->variant.float_kind);
    if (fb == tb) {
      v.ty = to;
      return v;
    }
    long double ld;
    if (fb == 4) {
      float f;
      std::memcpy(&f, &v.float_bits, 4);
      ld = f;
    } else if (fb == 8) {
      double d;
      std::memcpy(&d, &v.float_bits, 8);
      ld = d;
    } else {
      std::memcpy(&ld, &v.float_bits, 16);
    }
    if (tb == 4) {
      float f = (float)ld;
      unsigned __int128 bits = 0;
      std::memcpy(&bits, &f, 4);
      return float_const_bits(to->variant.float_kind, bits);
    }
    if (tb == 8) {
      double d = (double)ld;
      unsigned __int128 bits = 0;
      std::memcpy(&bits, &d, 8);
      return float_const_bits(to->variant.float_kind, bits);
    }
    unsigned __int128 bits = 0;
    std::memcpy(&bits, &ld, 16);
    return float_const_bits(to->variant.float_kind, bits);
  }
  if (to->kind == (a_type_kind)tk_integer) {
    long double ld;
    unsigned fb = float_kind_bytes(from->variant.float_kind);
    if (fb == 4) {
      float f;
      std::memcpy(&f, &v.float_bits, 4);
      ld = f;
    } else if (fb == 8) {
      double d;
      std::memcpy(&d, &v.float_bits, 8);
      ld = d;
    } else {
      std::memcpy(&ld, &v.float_bits, 16);
    }
    long long result = il_int_signed(to) ? (long long)ld
                                         : (long long)(unsigned long long)ld;
    return const_int_val(result, to);
  }
  unsupported_f("constant conversion %s -> %s", ty_name(from).c_str(),
                ty_name(to).c_str());
}

Val FuncEmitter::double_const_to(a_type_ptr to, double d) {
  a_float_kind kind = skip_typerefs(to)->variant.float_kind;
  unsigned __int128 bits = 0;
  if (kind == (a_float_kind)fk_float ||
      kind == (a_float_kind)fk_std_float32) {
    float f = (float)d;
    std::memcpy(&bits, &f, 4);
  } else if (kind == (a_float_kind)fk_double ||
             kind == (a_float_kind)fk_std_float64) {
    std::memcpy(&bits, &d, 8);
  } else {
    long double ld = (long double)d;
    std::memcpy(&bits, &ld, 16);
  }
  return float_const_bits(kind, bits);
}

Val FuncEmitter::convert(Val v, a_type_ptr from_il, a_type_ptr to_il) {
  a_type_ptr from = skip_typerefs(from_il);
  a_type_ptr to = skip_typerefs(to_il);
  if (ty_name(from) == ty_name(to)) return v;
  if (to->kind == (a_type_kind)tk_void) return v;
  if (from->kind == (a_type_kind)tk_array &&
      to->kind == (a_type_kind)tk_pointer) {
    // Array-to-pointer decay: the value already is the address.
    return v;
  }
  if (type_is_aggregate(from) || type_is_aggregate(to)) {
    unsupported_f("cast involving aggregate type (%s -> %s)",
                  ty_name(from).c_str(), ty_name(to).c_str());
  }
  if (v.const_kind != Val::NotConst) {
    return convert_const(v, from, to);
  }
  if (from->kind == (a_type_kind)tk_pointer &&
      to->kind == (a_type_kind)tk_pointer) {
    int n = new_value();
    ensure_open();
    emit("%%%d:%s = bitcast %%%d", n, ty_name(to).c_str(), v.n);
    return val(n, to);
  }
  if (from->kind == (a_type_kind)tk_pointer &&
      to->kind == (a_type_kind)tk_integer) {
    int n = new_value();
    ensure_open();
    emit("%%%d:%s = ptrtoint %%%d", n, ty_name(to).c_str(), v.n);
    return val(n, to);
  }
  if (from->kind == (a_type_kind)tk_integer &&
      to->kind == (a_type_kind)tk_pointer) {
    int n = new_value();
    ensure_open();
    emit("%%%d:%s = inttoptr %%%d", n, ty_name(to).c_str(), v.n);
    return val(n, to);
  }
  if (from->kind == (a_type_kind)tk_integer &&
      to->kind == (a_type_kind)tk_integer) {
    if (type_is_bool(from) && !type_is_bool(to)) {
      int n = new_value();
      ensure_open();
      emit("%%%d:%s = zext %%%d", n, ty_name(to).c_str(), v.n);
      return val(n, to);
    }
    unsigned fb = int_bits(from);
    unsigned tb = int_bits(to);
    if (tb < fb) {
      int n = new_value();
      ensure_open();
      emit("%%%d:%s = trunc %%%d", n, ty_name(to).c_str(), v.n);
      return val(n, to);
    }
    if (tb > fb) {
      const char *op = il_int_signed(from) ? "sext" : "zext";
      int n = new_value();
      ensure_open();
      emit("%%%d:%s = %s %%%d", n, ty_name(to).c_str(), op, v.n);
      return val(n, to);
    }
    return v;
  }
  if (from->kind == (a_type_kind)tk_integer &&
      to->kind == (a_type_kind)tk_float) {
    const char *op = il_int_signed(from) ? "sitof" : "uitof";
    int n = new_value();
    ensure_open();
    emit("%%%d:%s = %s %%%d", n, ty_name(to).c_str(), op, v.n);
    return val(n, to);
  }
  if (from->kind == (a_type_kind)tk_float &&
      to->kind == (a_type_kind)tk_integer) {
    const char *op = il_int_signed(to) ? "sftoi" : "uftoi";
    int n = new_value();
    ensure_open();
    emit("%%%d:%s = %s %%%d", n, ty_name(to).c_str(), op, v.n);
    return val(n, to);
  }
  if (from->kind == (a_type_kind)tk_float &&
      to->kind == (a_type_kind)tk_float) {
    unsigned fb = float_kind_bytes(from->variant.float_kind);
    unsigned tb = float_kind_bytes(to->variant.float_kind);
    const char *op = tb < fb ? "fptrunc" : "fpext";
    if (fb == tb) return v;
    int n = new_value();
    ensure_open();
    emit("%%%d:%s = %s %%%d", n, ty_name(to).c_str(), op, v.n);
    return val(n, to);
  }
  unsupported_f("conversion %s -> %s", ty_name(from).c_str(),
                ty_name(to).c_str());
}

// Binary arithmetic/comparison op dispatch, with the signedness taken
// from the IL operand type.
Val FuncEmitter::binop(an_expr_operator_kind op, Val a, Val b,
                       a_type_ptr result_ty, a_type_ptr operand_ty) {
  a_type_ptr res = skip_typerefs(result_ty);
  if (res->kind == (a_type_kind)tk_float) {
    const char *name;
    switch (op) {
      case eok_add: name = "fadd"; break;
      case eok_subtract: name = "fsub"; break;
      case eok_multiply: name = "fmul"; break;
      case eok_divide: name = "fdiv"; break;
      default: unsupported_f("float binop %d", (int)op);
    }
    ensure_open();
    int n = new_value();
    emit("%%%d:%s = %s %s, %s", n, ty_name(res).c_str(), name, opnd(a).c_str(),
         opnd(b).c_str());
    return val(n, res);
  }
  const char *name;
  bool signed_ops = il_int_signed(operand_ty);
  switch (op) {
    case eok_add: name = "add"; break;
    case eok_subtract: name = "sub"; break;
    case eok_multiply: name = "mul"; break;
    case eok_divide: name = signed_ops ? "sdiv" : "udiv"; break;
    case eok_remainder: name = signed_ops ? "srem" : "urem"; break;
    case eok_and: name = "and"; break;
    case eok_or: name = "or"; break;
    case eok_xor: name = "xor"; break;
    case eok_shiftl: name = "shl"; break;
    case eok_shiftr: name = signed_ops ? "ashr" : "lshr"; break;
    default: unsupported_f("integer binop %d", (int)op);
  }
  ensure_open();
  int n = new_value();
  emit("%%%d:%s = %s %s, %s", n, ty_name(res).c_str(), name, opnd(a).c_str(),
       opnd(b).c_str());
  return val(n, res);
}

Val FuncEmitter::field_base(an_expr_node_ptr obj) {
  a_type_ptr obj_ty = skip_typerefs(obj->type);
  if (obj->is_lvalue || obj->is_xvalue) {
    return place(obj);
  }
  if (type_is_aggregate(obj_ty)) {
    // Aggregate rvalue: its value IS an address; give it stable
    // storage so member selection has a place to work on.
    Val value = eval(obj);
    Val slot = alloca(obj_ty);
    memcpy_bytes(slot, value, (unsigned long long)obj_ty->size);
    return slot;
  }
  return place(obj);
}

Val FuncEmitter::place(an_expr_node_ptr expr) {
  switch (expr->kind) {
    case enk_variable:
      return var_storage(expr->variant.variable.ptr);
    case enk_constant: {
      a_constant_ptr con = expr->variant.constant.ptr;
      if (con->kind == (a_constant_repr_kind)ck_string) {
        return string_slot(con);
      }
      if (con->kind == (a_constant_repr_kind)ck_address) {
        a_constant_ptr addr = con;
        if (addr->variant.address.kind == (an_address_base_kind)abk_variable) {
          return var_storage(addr->variant.address.variant.variable);
        }
        if (addr->variant.address.kind == (an_address_base_kind)abk_routine) {
          unsupported("address-of-routine constant");
        }
        unsupported_f("address constant kind %d",
                      (int)addr->variant.address.kind);
      }
      unsupported_f("constant kind %d used as an lvalue", (int)con->kind);
    }
    case enk_operation: {
      an_expr_node_ptr op1 = expr->variant.operation.operands;
      switch (expr->variant.operation.kind) {
        case eok_indirect: {
          Val ptr = eval(op1);
          return val(ptr.n, synth_ptr_il_type(skip_typerefs(expr->type)));
        }
        case eok_dot_field:
        case eok_points_to_field: {
          Val base;
          bool erased_base = false;
          if (expr->variant.operation.kind == (an_expr_operator_kind)eok_dot_field) {
            base = field_base(op1);
          } else {
            erased_base = base_pointee_erased(op1->type);
            base = eval(op1);
          }
          a_field_ptr field = op1->next->variant.field.ptr;
          if (erased_base) {
            return byte_offset_addr(base, (unsigned long long)field->offset,
                                    field->type);
          }
          int index = 0;
          a_type_ptr cls = skip_typerefs(
              expr->variant.operation.kind == (an_expr_operator_kind)eok_dot_field
                  ? op1->type
                  : (skip_typerefs(op1->type)->variant.pointer.type));
          for (a_field_ptr f = cls->variant.class_struct_union.field_list;
               f != NULL && f != field; f = f->next) {
            index++;
          }
          return gep(base, field->type, "field " + std::to_string(index));
        }
        case eok_subscript: {
          a_type_ptr base_ty = skip_typerefs(op1->type);
          Val idx = eval(op1->next);
          bool idx_const =
              idx.const_kind == Val::IntC && idx.int_val >= 0;
          if (base_ty->kind == (a_type_kind)tk_array) {
            Val base = place(op1);
            if (idx_const) {
              return gep(base, skip_typerefs(expr->type),
                         std::to_string(idx.int_val));
            }
            return gep(base, skip_typerefs(expr->type),
                       "index %" + std::to_string(idx.n));
          }
          if (!idx_const) {
            Val base = eval(op1);
            return gep(base, skip_typerefs(expr->type),
                       "index %" + std::to_string(idx.n));
          }
          // A negative constant index goes through scaled byte
          // arithmetic (ConstIndex steps are unsigned).
          a_targ_size_t elem_size =
              base_pointee_erased(op1->type)
                  ? skip_typerefs(base_ty->variant.pointer.type)->size
                  : skip_typerefs(expr->type)->size;
          Val base = base_ty->kind == (a_type_kind)tk_array ? place(op1)
                                                            : eval(op1);
          a_type i64t = synth_int_type(64, TRUE);
          if (idx.int_val < 0) {
            Val wide = convert(idx, idx.ty, &i64t);
            Val base_int = convert(base, base.ty, &i64t);
            ensure_open();
            int n = new_value();
            emit("%%%d:i64 = add %s, %s", n, opnd(base_int).c_str(),
                 opnd(wide).c_str());
            Val sum = val(n, &i64t);
            ensure_open();
            int m = new_value();
            emit("%%%d:ptr(%s) = inttoptr %s", m, ty_name(expr->type).c_str(),
                 opnd(sum).c_str());
            return val(m, synth_ptr_il_type(skip_typerefs(expr->type)));
          }
          if (base_pointee_erased(op1->type)) {
            // The pointee erased to void: gep cannot stride it; scale
            // bytes explicitly.
            return scaled_byte_addr(base, idx, (unsigned long long)elem_size,
                                    expr->type);
          }
          // GEP strides by the element size for both array and
          // pointer bases.
          return gep(base, skip_typerefs(expr->type),
                     std::to_string(idx.int_val));
        }
        case eok_array_to_pointer: {
          a_type_ptr op_ty = skip_typerefs(op1->type);
          if (op_ty->kind == (a_type_kind)tk_array) {
            Val base = place(op1);
            return gep(base, op_ty->variant.array.element_type, "0");
          }
          return eval(op1);
        }
        case eok_lvalue_cast:
        case eok_lvalue_adjust:
          return place(op1);
        case eok_address_of:
          return place(op1);
        default:
          unsupported_f("operation %d used as an lvalue",
                        (int)expr->variant.operation.kind);
      }
    }
    case enk_param_ref: {
      unsigned num = expr->variant.param_ref.param_num;
      if (num == 0 || num > params.size()) {
        unsupported_f("param_ref %u out of range", num);
      }
      return var_storage(params[num - 1]);
    }
    default:
      unsupported_f("expression kind %d used as an lvalue", (int)expr->kind);
  }
}

Val FuncEmitter::eval(an_expr_node_ptr expr) {
  switch (expr->kind) {
    case enk_constant: {
      a_constant_ptr con = expr->variant.constant.ptr;
      switch (con->kind) {
        case ck_integer:
          return const_int_val(int_const_value(con), con->type);
        case ck_float: {
          unsigned __int128 bits = 0;
          std::memcpy(&bits, con->variant.float_value.bytes,
                      (size_t)float_kind_bytes(
                          skip_typerefs(con->type)->variant.float_kind));
          return float_const_bits(skip_typerefs(con->type)->variant.float_kind,
                                  bits);
        }
        case ck_string: {
          // A string literal is an array glvalue; its value is the
          // address of the synthesized global.
          Val slot = string_slot(con);
          if (expr->is_lvalue || expr->is_xvalue) return slot;
          return gep(slot,
                     skip_typerefs(string_array_il_type(con))
                         ->variant.array.element_type,
                     "0");
        }
        case ck_address: {
          Val base;
          if (con->variant.address.kind == (an_address_base_kind)abk_variable) {
            base = var_storage(con->variant.address.variant.variable);
          } else if (con->variant.address.kind ==
                     (an_address_base_kind)abk_routine) {
            base = call_addr(con->variant.address.variant.routine);
          } else if (con->variant.address.kind ==
                     (an_address_base_kind)abk_constant) {
            a_constant_ptr inner = con->variant.address.variant.constant;
            if (inner->kind != (a_constant_repr_kind)ck_string) {
              unsupported("address-of-nonstring constant");
            }
            base = string_slot(inner);
          } else {
            unsupported_f("address constant kind %d",
                          (int)con->variant.address.kind);
          }
          if (con->variant.address.offset != 0) {
            a_type i64 = synth_int_type(64, TRUE);
            Val as_int = convert(base, base.ty, &i64);
            Val off = int_const((long long)con->variant.address.offset, &i64);
            ensure_open();
            int n = new_value();
            emit("%%%d:i64 = add %%%d, %%%d", n, as_int.n, off.n);
            Val sum = val(n, &i64);
            return convert(sum, &i64, expr->type);
          }
          return convert(base, base.ty, expr->type);
        }
        default:
          unsupported_f("constant kind %d in an expression", (int)con->kind);
      }
    }
    case enk_variable: {
      a_variable_ptr var = expr->variant.variable.ptr;
      bool glvalue = expr->is_lvalue || expr->is_xvalue;
      if (glvalue) return var_storage(var);
      // Scalar prvalue: load. Aggregate prvalue: the loaded
      // aggregate-typed value is the address of a copy.
      return load(var_storage(var), expr->type);
    }
    case enk_param_ref: {
      unsigned num = expr->variant.param_ref.param_num;
      if (num == 0 || num > params.size()) {
        unsupported_f("param_ref %u out of range", num);
      }
      a_variable_ptr var = params[num - 1];
      bool glvalue = expr->is_lvalue || expr->is_xvalue;
      if (glvalue) return var_storage(var);
      return load(var_storage(var), expr->type);
    }
    case enk_routine:
      return call_addr(expr->variant.routine.ptr);
    case enk_operation:
      break; // handled below
    case enk_temp_init:
      unsupported("enk_temp_init survived lowering");
    default:
      unsupported_f("expression kind %d", (int)expr->kind);
  }

  an_expr_operator_kind op = expr->variant.operation.kind;
  an_expr_node_ptr op1 = expr->variant.operation.operands;
  an_expr_node_ptr op2 = op1 != NULL ? op1->next : NULL;
  an_expr_node_ptr op3 = op2 != NULL ? op2->next : NULL;
  a_type_ptr result_ty = expr->type;

  switch (op) {
    case eok_add:
    case eok_subtract:
    case eok_multiply:
    case eok_divide:
    case eok_remainder:
    case eok_and:
    case eok_or:
    case eok_xor:
    case eok_shiftl:
    case eok_shiftr: {
      Val a = eval(op1);
      Val b = eval(op2);
      return binop(op, a, b, result_ty, op1->type);
    }
    case eok_negate: {
      Val a = eval(op1);
      a_type_ptr ty = skip_typerefs(result_ty);
      const char *name =
          ty->kind == (a_type_kind)tk_float ? "fneg" : "neg";
      int n = new_value();
      ensure_open();
      emit("%%%d:%s = %s %s", n, ty_name(ty).c_str(), name, opnd(a).c_str());
      return val(n, ty);
    }
    case eok_complement: {
      Val a = eval(op1);
      int n = new_value();
      ensure_open();
      emit("%%%d:%s = not %s", n, ty_name(result_ty).c_str(),
           opnd(a).c_str());
      return val(n, result_ty);
    }
    case eok_unary_plus:
      return eval(op1);
    case eok_not: {
      Val b = to_bool(eval(op1));
      int n = new_value();
      ensure_open();
      emit("%%%d:bool = not %s", n, opnd(b).c_str());
      a_type bty = synth_bool_type();
      return convert(val(n, &bty), &bty, result_ty);
    }
    case eok_cast:
      return convert(eval(op1), op1->type, result_ty);
    case eok_lvalue_cast:
    case eok_lvalue_adjust:
      // An lvalue-preserving conversion; the address is unchanged.
      return place(expr);
    case eok_address_of: {
      if (op1->kind == (an_expr_node_kind)enk_routine) {
        return call_addr(op1->variant.routine.ptr);
      }
      return place(op1);
    }
    case eok_indirect: {
      // *p as an rvalue: load through the pointer (aggregates load as
      // their address-typed value).
      Val ptr = eval(op1);
      return load(ptr, result_ty);
    }
    case eok_dot_field:
    case eok_points_to_field: {
      Val addr = place(expr);
      if (expr->is_lvalue || expr->is_xvalue) return addr;
      return load(addr, result_ty);
    }
    case eok_subscript: {
      Val addr = place(expr);
      if (expr->is_lvalue || expr->is_xvalue) return addr;
      return load(addr, result_ty);
    }
    case eok_array_to_pointer:
      return place(expr);
    case eok_assign: {
      Val dst = place(op1);
      Val src = eval(op2);
      a_type_ptr ty = skip_typerefs(result_ty);
      if (type_is_aggregate(ty)) {
        memcpy_bytes(dst, src, (unsigned long long)ty->size);
      } else {
        store(dst, src);
      }
      return src;
    }
    case eok_add_assign:
    case eok_subtract_assign:
    case eok_multiply_assign:
    case eok_divide_assign:
    case eok_remainder_assign:
    case eok_and_assign:
    case eok_or_assign:
    case eok_xor_assign:
    case eok_shiftl_assign:
    case eok_shiftr_assign: {
      Val dst = place(op1);
      Val old = load(dst, op1->type);
      Val rhs = eval(op2);
      static const an_expr_operator_kind plain[] = {
          eok_add,    eok_subtract, eok_multiply, eok_divide,
          eok_remainder, eok_and,     eok_or,       eok_xor,
          eok_shiftl, eok_shiftr};
      int idx = (int)op - (int)eok_add_assign;
      Val newv = binop(plain[idx], old, rhs, op1->type, op1->type);
      store(dst, newv);
      return newv;
    }
    case eok_padd_assign:
    case eok_psubtract_assign: {
      Val dst = place(op1);
      Val old = load(dst, op1->type);
      Val idx = eval(op2);
      Val newv = pointer_arith(
          op == (an_expr_operator_kind)eok_padd_assign ? eok_padd : eok_psubtract,
          old, idx, op1->type);
      store(dst, newv);
      return newv;
    }
    case eok_pre_incr:
    case eok_pre_decr:
    case eok_post_incr:
    case eok_post_decr: {
      Val dst = place(op1);
      Val old = load(dst, op1->type);
      bool incr = op == (an_expr_operator_kind)eok_pre_incr ||
                  op == (an_expr_operator_kind)eok_post_incr;
      Val newv = incr ? increment(old, TRUE) : increment(old, FALSE);
      store(dst, newv);
      return op == (an_expr_operator_kind)eok_pre_incr ||
                     op == (an_expr_operator_kind)eok_pre_decr
                 ? newv
                 : old;
    }
    case eok_padd:
    case eok_psubtract: {
      Val base = eval(op1);
      Val idx = eval(op2);
      return pointer_arith(op, base, idx, result_ty);
    }
    case eok_pdiff: {
      a_type i64 = synth_int_type(64, TRUE);
      Val a = convert(eval(op1), op1->type, &i64);
      Val b = convert(eval(op2), op2->type, &i64);
      ensure_open();
      int n = new_value();
      emit("%%%d:i64 = sub %%%d, %%%d", n, a.n, b.n);
      Val diff = val(n, &i64);
      a_type_ptr pty = skip_typerefs(op1->type);
      a_targ_size_t elem_size = 1;
      if (pty->kind == (a_type_kind)tk_pointer) {
        elem_size = skip_typerefs(pty->variant.pointer.type)->size;
      }
      Val size = int_const((long long)elem_size, &i64);
      ensure_open();
      int m = new_value();
      emit("%%%d:i64 = sdiv %s, %s", m, opnd(diff).c_str(),
           opnd(size).c_str());
      return convert(val(m, &i64), &i64, result_ty);
    }
    case eok_eq:
    case eok_ne:
    case eok_lt:
    case eok_le:
    case eok_gt:
    case eok_ge: {
      Val a = eval(op1);
      Val b = eval(op2);
      a_type_ptr cmp_ty = skip_typerefs(op1->type);
      if (cmp_ty->kind == (a_type_kind)tk_float) {
        const char *pred;
        switch (op) {
          case eok_eq: pred = "eq"; break;
          case eok_ne: pred = "ne"; break;
          case eok_lt: pred = "lt"; break;
          case eok_le: pred = "le"; break;
          case eok_gt: pred = "gt"; break;
          default: pred = "ge"; break;
        }
        return fcmp(pred, a, b, result_ty);
      }
      if (cmp_ty->kind == (a_type_kind)tk_pointer) {
        a_type i64 = synth_int_type(64, TRUE);
        a = convert(a, cmp_ty, &i64);
        b = convert(b, op2->type, &i64);
        cmp_ty = &i64;
      }
      const char *pred;
      bool sign = il_int_signed(cmp_ty);
      switch (op) {
        case eok_eq: pred = "eq"; break;
        case eok_ne: pred = "ne"; break;
        case eok_lt: pred = sign ? "slt" : "ult"; break;
        case eok_le: pred = sign ? "sle" : "ule"; break;
        case eok_gt: pred = sign ? "sgt" : "ugt"; break;
        default: pred = sign ? "sge" : "uge"; break;
      }
      return icmp(pred, a, b, result_ty);
    }
    case eok_land:
    case eok_lor: {
      Val a = to_bool(eval(op1));
      a_type bty = synth_bool_type();
      Val slot = alloca(&bty);
      store(slot, a);
      int b_rhs = new_block();
      int b_end = new_block();
      if (op == (an_expr_operator_kind)eok_land) {
        emit_brif(a, b_rhs, -1, b_end, -1);
      } else {
        emit_brif(a, b_end, -1, b_rhs, -1);
      }
      start_block(b_rhs);
      Val b = to_bool(eval(op2));
      store(slot, b);
      start_block(b_end);
      Val res = load(slot, &bty);
      return convert(res, &bty, result_ty);
    }
    case eok_question: {
      Val cond = to_bool(eval(op1));
      bool has_value = skip_typerefs(result_ty)->kind != (a_type_kind)tk_void;
      Val slot;
      if (has_value) slot = alloca(result_ty);
      int b_then = new_block();
      int b_else = new_block();
      int b_end = new_block();
      emit_brif(cond, b_then, -1, b_else, -1);
      start_block(b_then);
      if (has_value) {
        Val tv = convert(eval(op2), op2->type, result_ty);
        store_or_copy(slot, tv, result_ty);
      }
      emit_br(b_end);
      start_block(b_else);
      if (has_value) {
        Val ev = convert(eval(op3), op3->type, result_ty);
        store_or_copy(slot, ev, result_ty);
      }
      emit_br(b_end);
      start_block(b_end);
      if (!has_value) return val(-1, result_ty);
      return load(slot, result_ty);
    }
    case eok_comma:
      eval(op1);
      return eval(op2);
    case eok_call:
      return call_emit(op1, op1->next, result_ty);
    default:
      unsupported_f("expression operation %d", (int)op);
  }
}

// Process-lifetime IL types for values the IL does not carry: void
// and a pointer to it (the MIR spelling of a function address).
a_type_ptr static_void_type() {
  static a_type v = []() {
    a_type t;
    std::memset(&t, 0, sizeof(t));
    t.kind = (a_type_kind)tk_void;
    return t;
  }();
  return &v;
}

a_type_ptr static_funcptr_type() {
  static a_type p = []() {
    a_type t;
    std::memset(&t, 0, sizeof(t));
    t.kind = (a_type_kind)tk_pointer;
    t.variant.pointer.type = static_void_type();
    return t;
  }();
  return &p;
}

// Address of a routine (funcaddr when defined in this TU, externaddr
// otherwise). Function pointers carry no signature in their MIR type.
Val FuncEmitter::call_addr(a_routine_ptr rout) {
  std::string name = entity_name(&rout->source_corresp, "routine");
  bool defined = rout->function_def_number != NULL_function_def_number;
  ensure_open();
  int n = new_value();
  if (defined) {
    emit("%%%d:ptr(void) = funcaddr @%s", n, name.c_str());
  } else {
    emit("%%%d:ptr(void) = externaddr @%s", n, name.c_str());
  }
  return val(n, static_funcptr_type());
}

// +1 / -1 on a scalar or pointer value.
Val FuncEmitter::increment(Val v, bool up) {
  a_type_ptr ty = skip_typerefs(v.ty);
  if (ty->kind == (a_type_kind)tk_pointer) {
    a_type i64 = synth_int_type(64, TRUE);
    Val as_int = convert(v, ty, &i64);
    a_targ_size_t step = skip_typerefs(ty->variant.pointer.type)->size;
    Val off = int_const(up ? (long long)step : -(long long)step, &i64);
    ensure_open();
    int n = new_value();
    emit("%%%d:i64 = add %s, %s", n, opnd(as_int).c_str(), opnd(off).c_str());
    Val sum = val(n, &i64);
    return convert(sum, &i64, ty);
  }
  Val one = int_const(up ? 1 : -1, ty);
  return binop(eok_add, v, one, ty, ty);
}

// Pointer +/- integer, scaled by the pointee size.
Val FuncEmitter::pointer_arith(an_expr_operator_kind op, Val base, Val idx,
                               a_type_ptr result_ty) {
  a_type_ptr base_ty = skip_typerefs(base.ty);
  a_type_ptr elem = base_ty->kind == (a_type_kind)tk_pointer
                        ? base_ty->variant.pointer.type
                        : base_ty;
  if (op == (an_expr_operator_kind)eok_padd) {
    if (idx.const_kind == Val::IntC && idx.int_val < 0) {
      Val mag = const_int_val(-idx.int_val, idx.ty);
      return pointer_arith((an_expr_operator_kind)eok_psubtract, base, mag,
                           result_ty);
    }
    if (base_pointee_erased(base.ty)) {
      return scaled_byte_addr(base, idx, (unsigned long long)elem->size, elem);
    }
    bool is_const;
    return gep(base, elem, gep_step(idx, &is_const));
  }
  a_type i64 = synth_int_type(64, TRUE);
  Val as_int = convert(base, base_ty, &i64);
  Val wide = convert(idx, idx.ty, &i64);
  ensure_open();
  int n = new_value();
  if (wide.const_kind != Val::NotConst) {
    // A constant index folds: base - idx*scale as one address add.
    long long scale = (long long)skip_typerefs(elem)->size;
    Val off = const_int_val(-wide.int_val * scale, &i64);
    emit("%%%d:i64 = add %s, %s", n, opnd(as_int).c_str(), opnd(off).c_str());
  } else {
    emit("%%%d:i64 = neg %%%d", n, wide.n);
    Val neg = val(n, &i64);
    ensure_open();
    int m = new_value();
    emit("%%%d:i64 = add %s, %s", m, opnd(as_int).c_str(), opnd(neg).c_str());
    n = m;
  }
  Val sum = val(n, &i64);
  return convert(sum, &i64, result_ty);
}

Val FuncEmitter::call_emit(an_expr_node_ptr selector, an_expr_node_ptr args,
                           a_type_ptr result_ty) {
  std::vector<Val> argv;
  std::string args_text;
  an_expr_node_ptr arg_node = args;
  for (an_expr_node_ptr a = args; a != NULL; a = a->next) {
    argv.push_back(eval(a));
  }
  // Argument types come from the IL argument nodes: a decayed array
  // argument spells ptr(elem), an aggregate argument spells its
  // aggregate type (the value is the address of the storage).
  for (size_t i = 0; i < argv.size(); i++, arg_node = arg_node->next) {
    if (i > 0) args_text += ", ";
    args_text += ty_name(arg_node->type) + " " + opnd(argv[i]);
  }
  ensure_open();
  if (selector->kind == (an_expr_node_kind)enk_routine) {
    a_routine_ptr rout = selector->variant.routine.ptr;
    std::string name = entity_name(&rout->source_corresp, "routine");
    bool defined = rout->function_def_number != NULL_function_def_number;
    const char *kw = defined ? "call" : "call.ext";
    if (skip_typerefs(result_ty)->kind == (a_type_kind)tk_void) {
      emit("%s @%s %s", kw, name.c_str(), args_text.c_str());
      return val(-1, result_ty);
    }
    int n = new_value();
    emit("%%%d:%s = %s @%s %s", n, ty_name(result_ty).c_str(), kw,
         name.c_str(), args_text.c_str());
    return val(n, result_ty);
  }
  // Indirect call through a function pointer.
  Val fp = eval(selector);
  a_type_ptr pty = skip_typerefs(selector->type);
  if (pty->kind == (a_type_kind)tk_pointer) {
    pty = skip_typerefs(pty->variant.pointer.type);
  }
  if (pty->kind != (a_type_kind)tk_routine) {
    unsupported("indirect call through a non-function pointer");
  }
  a_routine_type_supplement_ptr extra = pty->variant.routine.extra_info;
  if (extra == NULL || !extra->prototyped || extra->has_ellipsis) {
    unsupported("indirect call through a variadic function pointer");
  }
  std::string sig = "sig=(";
  bool first = TRUE;
  for (a_param_type_ptr param = extra->param_type_list; param != NULL;
       param = param->next) {
    if (!first) sig += ", ";
    first = FALSE;
    sig += ty_name(param->type);
  }
  sig += ") -> " + ty_name(pty->variant.routine.return_type);
  if (skip_typerefs(result_ty)->kind == (a_type_kind)tk_void) {
    emit("call.ind %s %%%d, %s", sig.c_str(), fp.n, args_text.c_str());
    return val(-1, result_ty);
  }
  int n = new_value();
  emit("%%%d:%s = call.ind %s %%%d, %s", n, ty_name(result_ty).c_str(),
       sig.c_str(), fp.n, args_text.c_str());
  return val(n, result_ty);
}

// Writes a constant into storage (used for local constant
// initializers). A NULL constant means zero fill.
void FuncEmitter::store_const_into(Val addr, a_constant_ptr con,
                                   a_type_ptr type) {
  a_type_ptr ty = skip_typerefs(type);
  if (con == NULL) {
    ensure_open();
    emit("memset %%%d, byte=0, len=%llu", addr.n, (unsigned long long)ty->size);
    return;
  }
  switch (con->kind) {
    case ck_integer:
      store(addr, const_int_val(int_const_value(con), con->type));
      return;
    case ck_float: {
      unsigned __int128 bits = 0;
      std::memcpy(&bits, con->variant.float_value.bytes,
                  (size_t)float_kind_bytes(
                      skip_typerefs(con->type)->variant.float_kind));
      store(addr, float_const_bits(
                      skip_typerefs(con->type)->variant.float_kind, bits));
      return;
    }
    case ck_string: {
      a_type_ptr arr_ty = string_array_il_type(con);
      a_type_ptr elem_ty =
          skip_typerefs(arr_ty)->variant.array.element_type;
      for (a_targ_size_t i = 0; i < con->variant.string.length; i++) {
        Val byte_addr = gep(addr, elem_ty, std::to_string((unsigned long long)i));
        Val byte = int_const((unsigned char)con->variant.string.value[i],
                             elem_ty);
        store(byte_addr, byte);
      }
      return;
    }
    case ck_aggregate: {
      std::vector<a_constant_ptr> elems;
      for (a_constant_ptr elem = con->variant.aggregate.first_constant;
           elem != NULL; elem = elem->next) {
        elems.push_back(elem);
      }
      if (ty->kind == (a_type_kind)tk_array) {
        a_type_ptr elem_ty = ty->variant.array.element_type;
        a_targ_size_t count = ty->variant.array.variant.number_of_elements;
        for (a_targ_size_t i = 0; i < count; i++) {
          Val elem_addr =
              gep(addr, elem_ty, std::to_string((unsigned long long)i));
          store_const_into(elem_addr, i < elems.size() ? elems[i] : NULL,
                           elem_ty);
        }
        return;
      }
      if (ty->kind == (a_type_kind)tk_struct ||
          ty->kind == (a_type_kind)tk_class ||
          ty->kind == (a_type_kind)tk_union) {
        size_t i = 0;
        for (a_field_ptr f = ty->variant.class_struct_union.field_list;
             f != NULL; f = f->next, i++) {
          Val field_addr = gep(addr, f->type,
                               "field " + std::to_string((unsigned long long)i));
          store_const_into(field_addr, i < elems.size() ? elems[i] : NULL,
                           f->type);
        }
        return;
      }
      unsupported_f("aggregate constant of type kind %d", (int)ty->kind);
    }
    default:
      unsupported_f("constant kind %d stored into a local", (int)con->kind);
  }
}

void FuncEmitter::do_dynamic_init(a_dynamic_init_ptr dip) {
  if (dip == NULL || inits_done_at_decl.count(dip)) return;
  a_variable_ptr var = dip->variable;
  switch (dip->kind) {
    case dik_none:
      break;
    case dik_expression: {
      Val value = eval(dip->variant.expression);
      Val slot = var_storage(var);
      if (type_is_aggregate(skip_typerefs(var->type))) {
        memcpy_bytes(slot, value, (unsigned long long)var->type->size);
      } else {
        store(slot, value);
      }
      break;
    }
    case dik_constant:
    case dik_nonconstant_aggregate:
      store_const_into(var_storage(var), dip->variant.constant.ptr, var->type);
      break;
    default:
      unsupported_f("dynamic init kind %d", (int)dip->kind);
  }
}

// Allocas (and constant initializers) for a block's automatic locals.
// Locals live on the associated scope's nonstatic_variables list, not
// on stmk_decl statements -- mirrors c_gen_be's dump_block_declarations.
void FuncEmitter::declare_block_locals(a_statement_ptr block) {
  a_scope_ptr scope = NULL;
  if (func_scope != NULL && func_scope->assoc_block == block) {
    scope = func_scope;
  } else if (block->variant.block.extra_info != NULL) {
    scope = block->variant.block.extra_info->assoc_scope;
  }
  if (scope == NULL) return;
  for (a_variable_ptr var = scope->nonstatic_variables; var != NULL;
       var = var->next) {
    if (ignore_variable_in_back_end(var)) continue;
    Val slot = alloca(var->type);
    slots.emplace(var, slot.n);
    if (var->init_kind == (an_init_kind)initk_zero) {
      store_const_into(slot, NULL, var->type);
    } else if (var->init_kind == (an_init_kind)initk_dynamic) {
      a_dynamic_init_ptr dip = var->initializer.dynamic;
      if (dip->kind == (a_dynamic_init_kind)dik_constant &&
          !dip->follows_an_exec_statement) {
        store_const_into(slot, dip->variant.constant.ptr, var->type);
        inits_done_at_decl.emplace(dip, true);
      }
    }
  }
}

int FuncEmitter::label_block(a_label_ptr label) {
  auto it = label_blocks.find(label);
  if (it != label_blocks.end()) return it->second;
  int b = new_block();
  label_blocks.emplace(label, b);
  return b;
}

void FuncEmitter::walk_statements(a_statement_ptr stmt) {
  for (; stmt != NULL; stmt = stmt->next) {
    walk_statement(stmt);
  }
}

void FuncEmitter::walk_statement(a_statement_ptr stmt) {
  switch (stmt->kind) {
    case stmk_block:
      declare_block_locals(stmt);
      walk_statements(stmt->variant.block.statements);
      break;
    case stmk_expr:
      eval(stmt->expr);
      break;
    case stmk_init:
      do_dynamic_init(stmt->variant.dynamic_init);
      break;
    case stmk_if: {
      Val cond = to_bool(eval(stmt->expr));
      a_statement_ptr then_stmt = stmt->variant.if_stmt.then_statement;
      a_statement_ptr else_stmt = stmt->variant.if_stmt.else_statement;
      int b_then = new_block();
      int b_end = new_block();
      if (else_stmt != NULL) {
        int b_else = new_block();
        emit_brif(cond, b_then, -1, b_else, -1);
        start_block(b_then);
        walk_statement(then_stmt);
        emit_br(b_end);
        start_block(b_else);
        walk_statement(else_stmt);
      } else {
        emit_brif(cond, b_then, -1, b_end, -1);
        start_block(b_then);
        walk_statement(then_stmt);
      }
      start_block(b_end);
      break;
    }
    case stmk_while: {
      int b_cond = new_block();
      int b_body = new_block();
      int b_end = new_block();
      emit_br(b_cond);
      start_block(b_cond);
      Val cond = to_bool(eval(stmt->expr));
      emit_brif(cond, b_body, -1, b_end, -1);
      start_block(b_body);
      walk_statement(stmt->variant.loop_statement);
      emit_br(b_cond);
      start_block(b_end);
      break;
    }
    case stmk_end_test_while: {
      int b_body = new_block();
      int b_cond = new_block();
      int b_end = new_block();
      emit_br(b_body);
      start_block(b_body);
      walk_statement(stmt->variant.loop_statement);
      emit_br(b_cond);
      start_block(b_cond);
      Val cond = to_bool(eval(stmt->expr));
      emit_brif(cond, b_body, -1, b_end, -1);
      start_block(b_end);
      break;
    }
    case stmk_for: {
      a_for_loop_ptr extra = stmt->variant.for_loop.extra_info;
      if (extra->initialization != NULL) {
        walk_statement(extra->initialization);
      }
      int b_cond = new_block();
      int b_body = new_block();
      int b_incr = new_block();
      int b_end = new_block();
      emit_br(b_cond);
      start_block(b_cond);
      if (stmt->expr != NULL) {
        Val cond = to_bool(eval(stmt->expr));
        emit_brif(cond, b_body, -1, b_end, -1);
      } else {
        a_type bty = synth_bool_type();
        Val one = int_const(1, &bty);
        emit_brif(one, b_body, -1, b_end, -1);
      }
      start_block(b_body);
      walk_statement(stmt->variant.for_loop.statement);
      emit_br(b_incr);
      start_block(b_incr);
      if (extra->increment != NULL) {
        eval(extra->increment);
      }
      emit_br(b_cond);
      start_block(b_end);
      break;
    }
    case stmk_switch: {
      Val v = eval(stmt->expr);
      bool scrut_const = v.const_kind == Val::IntC;
      long long scrut_val = scrut_const ? v.int_val : 0;
      // Collect the case labels in source order.
      std::vector<std::pair<long long, int>> cases;
      int default_block = -1;
      std::vector<a_statement_ptr> stack;
      stack.push_back(stmt->variant.switch_stmt.body_statement);
      while (!stack.empty()) {
        a_statement_ptr s = stack.back();
        stack.pop_back();
        for (; s != NULL; s = s->next) {
          if (s->kind == (a_statement_kind)stmk_switch_case) {
            a_switch_case_entry_ptr scep = s->variant.switch_case.extra_info;
            if (scep->range_end != NULL) {
              unsupported("GNU case range in switch");
            }
            int b = new_block();
            case_blocks.emplace(s, b);
            if (scep->case_value == NULL) {
              default_block = b;
            } else {
              cases.emplace_back(
                  std::strtoll(int_const_text(scep->case_value).c_str(),
                               NULL, 10),
                  b);
            }
          } else if (s->kind == (a_statement_kind)stmk_block) {
            stack.push_back(s->variant.block.statements);
          }
        }
      }
      bool has_explicit_default = default_block >= 0;
      if (scrut_const) {
        // A constant scrutinee never reaches the text as a switch
        // (the parser requires a value operand): branch to the
        // matching case (or default, or past the body when neither
        // exists).
        int target = -1;
        for (auto &c : cases) {
          if (c.first == scrut_val) {
            target = c.second;
            break;
          }
        }
        if (target < 0) target = default_block;
        if (target >= 0) {
          emit_br(target);
        } else {
          int b_skip = new_block();
          emit_br(b_skip);
          walk_statement(stmt->variant.switch_stmt.body_statement);
          start_block(b_skip);
          break;
        }
      } else {
        if (!has_explicit_default) default_block = new_block();
        ensure_open();
        std::string text = "  switch %" + std::to_string(v.n) + " [";
        bool first = TRUE;
        for (auto &c : cases) {
          if (!first) text += ", ";
          first = FALSE;
          text += std::to_string(c.first) + ": b" + std::to_string(c.second);
        }
        text += "] default b" + std::to_string(default_block);
        std::fputs(text.c_str(), f_out);
        std::fputc('\n', f_out);
        terminated = true;
        // A switch without an explicit default gets an empty
        // unreachable default block so every label has a terminator.
        if (!has_explicit_default) {
          start_block(default_block);
          emit("unreachable");
          terminated = true;
        }
      }
      walk_statement(stmt->variant.switch_stmt.body_statement);
      break;
    }
    case stmk_switch_case: {
      auto it = case_blocks.find(stmt);
      if (it != case_blocks.end()) start_block(it->second);
      break;
    }
    case stmk_return: {
      ensure_open();
      if (stmt->expr != NULL) {
        Val v = eval(stmt->expr);
        a_type_ptr ret_ty =
            skip_typerefs(func_rout->type)->variant.routine.return_type;
        if (skip_typerefs(ret_ty)->kind != (a_type_kind)tk_void) {
          if (!type_is_aggregate(skip_typerefs(ret_ty))) {
            v = convert(v, stmt->expr->type, ret_ty);
          }
          emit("ret %s", opnd(v).c_str());
        } else {
          emit("ret");
        }
      } else {
        emit("ret");
      }
      terminated = true;
      break;
    }
    case stmk_goto:
      emit_br(label_block(stmt->variant.label.ptr));
      break;
    case stmk_label:
      start_block(label_block(stmt->variant.label.ptr));
      break;
    case stmk_empty:
      break;
    case stmk_decl:
      // Locals are declared from the block's associated scope (see
      // declare_block_locals); the statement itself carries nothing.
      break;
    case stmk_try_block:
      unsupported("try block (exception handling)");
      break;
    case stmk_asm:
      unsupported("inline asm statement");
      break;
    default:
      unsupported_f("statement kind %d", (int)stmt->kind);
  }
}

// ---------------------------------------------------------------------------
// Module-level walk
// ---------------------------------------------------------------------------

void scan_const_strings(a_constant_ptr con);

void scan_expr_strings(an_expr_node_ptr expr) {
  for (; expr != NULL; expr = expr->next) {
    if (expr->kind == (an_expr_node_kind)enk_constant) {
      scan_const_strings(expr->variant.constant.ptr);
    } else if (expr->kind == (an_expr_node_kind)enk_operation) {
      scan_expr_strings(expr->variant.operation.operands);
    }
  }
}

void scan_const_strings(a_constant_ptr con) {
  if (con == NULL) return;
  if (con->kind == (a_constant_repr_kind)ck_string) {
    string_global_name(con);
  } else if (con->kind == (a_constant_repr_kind)ck_aggregate) {
    for (a_constant_ptr elem = con->variant.aggregate.first_constant;
         elem != NULL; elem = elem->next) {
      scan_const_strings(elem);
    }
  } else if (con->kind == (a_constant_repr_kind)ck_address &&
             con->variant.address.kind == (an_address_base_kind)abk_constant) {
    scan_const_strings(con->variant.address.variant.constant);
  }
}

void scan_stmt_strings(a_statement_ptr stmt) {
  for (; stmt != NULL; stmt = stmt->next) {
    switch (stmt->kind) {
      case stmk_block:
        scan_stmt_strings(stmt->variant.block.statements);
        break;
      case stmk_expr:
        scan_expr_strings(stmt->expr);
        break;
      case stmk_init: {
        a_dynamic_init_ptr dip = stmt->variant.dynamic_init;
        if (dip != NULL &&
            dip->kind == (a_dynamic_init_kind)dik_expression) {
          scan_expr_strings(dip->variant.expression);
        }
        break;
      }
      case stmk_if:
        scan_expr_strings(stmt->expr);
        scan_stmt_strings(stmt->variant.if_stmt.then_statement);
        scan_stmt_strings(stmt->variant.if_stmt.else_statement);
        break;
      case stmk_while:
      case stmk_end_test_while:
        scan_expr_strings(stmt->expr);
        scan_stmt_strings(stmt->variant.loop_statement);
        break;
      case stmk_for:
        if (stmt->variant.for_loop.extra_info->initialization != NULL) {
          scan_stmt_strings(stmt->variant.for_loop.extra_info->initialization);
        }
        scan_expr_strings(stmt->expr);
        if (stmt->variant.for_loop.extra_info->increment != NULL) {
          scan_expr_strings(stmt->variant.for_loop.extra_info->increment);
        }
        scan_stmt_strings(stmt->variant.for_loop.statement);
        break;
      case stmk_switch:
        scan_expr_strings(stmt->expr);
        scan_stmt_strings(stmt->variant.switch_stmt.body_statement);
        break;
      case stmk_return:
        scan_expr_strings(stmt->expr);
        break;
      default:
        break;
    }
  }
}

void emit_variable(a_variable_ptr var) {
  // Types not marked "needed" are not written to the IL file; their
  // read-back pointer lands on the shared error entry. That only
  // happens for front-end bookkeeping variables (EH region/stack
  // trackers), which v1 does not model.
  if (skip_typerefs(var->type)->kind == (a_type_kind)tk_error) {
    // The type entry did not survive the IL round trip (types are
    // written only when "needed"). That is fine for a declaration
    // nobody needed -- it is spelled ptr(void) for pointers anyway --
    // but never for a definition.
    if (var->storage_class == (a_storage_class)sc_extern) {
      std::fprintf(f_out, ".dataextern @%s ty=ptr(void)\n",
                   entity_name(&var->source_corresp, "variable").c_str());
      return;
    }
    unsupported_f("variable '%s' with an unwritten type",
                  var->source_corresp.name ? var->source_corresp.name : "?");
  }
  std::string name = entity_name(&var->source_corresp, "variable");
  if (var->is_thread_local) {
    unsupported_f("thread_local variable '%s'", name.c_str());
  }
  std::string ty = ty_name(var->type);
  if (var->storage_class == (a_storage_class)sc_extern) {
    if (!declared_globals.count(name)) {
      declared_globals.emplace(name, true);
      std::fprintf(f_out, ".dataextern @%s ty=%s\n", name.c_str(),
                   ty.c_str());
    }
    return;
  }
  std::string init = "zeroinit";
  switch (var->init_kind) {
    case initk_static:
      init = const_init_text(var->initializer.constant);
      break;
    case initk_zero:
    case initk_function_local:
      init = "zeroinit";
      break;
    case initk_dynamic: {
      a_dynamic_init_ptr dip = var->initializer.dynamic;
      if (dip != NULL &&
          dip->kind == (a_dynamic_init_kind)dik_constant &&
          !dip->follows_an_exec_statement) {
        init = const_init_text(dip->variant.constant.ptr);
      } else if (var->initialization_rewritten_as_assignment) {
        // The dynamic init runs in a generated __sti__ routine before
        // main; the storage starts zeroed.
        init = "zeroinit";
      } else {
        unsupported_f("dynamic initializer for global '%s'", name.c_str());
      }
      break;
    }
    case initk_none:
      init = "zeroinit";
      break;
    default:
      unsupported_f("init kind %d for variable '%s'", (int)var->init_kind,
                    name.c_str());
  }
  std::string attrs;
  if (var->storage_class == (a_storage_class)sc_static) attrs += " local";
  if (var->alignment > 1) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), " align=%llu",
                  (unsigned long long)var->alignment);
    attrs += buf;
  }
  if (declared_globals.count(name)) return;
  declared_globals.emplace(name, true);
  std::fprintf(f_out, ".global @%s%s ty=%s init=%s\n", name.c_str(),
               attrs.c_str(), ty.c_str(), init.c_str());
}

void emit_routine_sig(a_routine_ptr rout, std::string *out) {
  a_type_ptr type = skip_typerefs(rout->type);
  a_routine_type_supplement_ptr extra = type->variant.routine.extra_info;
  bool variadic = extra != NULL && extra->has_ellipsis;
  std::string sig = "sig=(";
  bool first = TRUE;
  if (extra != NULL && extra->prototyped) {
    for (a_param_type_ptr param = extra->param_type_list; param != NULL;
         param = param->next) {
      if (!first) sig += ", ";
      first = FALSE;
      sig += ty_name(param->type);
    }
  }
  if (variadic) {
    if (!first) sig += ", ";
    sig += "...";
  }
  sig += ") -> " + ty_name(type->variant.routine.return_type);
  *out = sig;
}

void emit_extern_routine(a_routine_ptr rout) {
  std::string sig;
  emit_routine_sig(rout, &sig);
  std::fprintf(f_out, ".extern @%s %s\n",
               entity_name(&rout->source_corresp, "routine").c_str(),
               sig.c_str());
}

void emit_func_def(a_routine_ptr rout, a_scope_ptr scope) {
  if (rout->special_kind != (a_special_function_kind)sfk_none) {
    // Constructors/destructors only appear once class support lands.
    unsupported_f("special member function '%s' (special kind %d)",
                  rout->source_corresp.name ? rout->source_corresp.name : "?",
                  (int)rout->special_kind);
  }
  if (scope->variant.routine.constructor_inits != NULL) {
    unsupported("constructor initializers");
  }
  FuncEmitter em;
  em.func_scope = scope;
  em.func_rout = rout;
  for (a_variable_ptr param = scope->variant.routine.parameters;
       param != NULL; param = param->next) {
    em.params.push_back(param);
  }
  std::string attrs;
  if (rout->storage_class == (a_storage_class)sc_static) attrs += " local";
  if (rout->is_inline) attrs += " inline";
  std::string sig;
  emit_routine_sig(rout, &sig);
  std::fprintf(f_out, ".func @%s%s %s\n",
               entity_name(&rout->source_corresp, "routine").c_str(),
               attrs.c_str(), sig.c_str());
  if (!em.params.empty()) {
    std::fprintf(f_out, ".params ");
    for (size_t i = 0; i < em.params.size(); i++) {
      if (i > 0) std::fprintf(f_out, ", ");
      std::fprintf(f_out, "%%%zu:%s", i, ty_name(em.params[i]->type).c_str());
    }
    std::fprintf(f_out, "\n");
    // Entry-block parameter values occupy %0..%n-1.
    em.next_value = (int)em.params.size();
  }
  // -O0-style spill: every parameter gets a writable slot; scalar
  // params store, aggregate params (whose value is the address of the
  // caller's copy) are memcpy'd so the callee owns its storage.
  for (size_t i = 0; i < em.params.size(); i++) {
    a_variable_ptr param = em.params[i];
    Val pv = em.val((int)i, param->type);
    Val slot = em.alloca(param->type);
    em.slots.emplace(param, slot.n);
    if (type_is_aggregate(skip_typerefs(param->type))) {
      em.memcpy_bytes(slot, pv, (unsigned long long)param->type->size);
    } else {
      em.store(slot, pv);
    }
  }
  em.walk_statement(scope->assoc_block);
  if (!em.terminated) {
    a_type_ptr ret_ty = skip_typerefs(rout->type)->variant.routine.return_type;
    if (skip_typerefs(ret_ty)->kind == (a_type_kind)tk_void) {
      em.emit("ret");
    } else {
      em.emit("ret undef");
    }
  }
}

void flush_pending_strings() {
  for (const std::string &line : pending_string_lines) {
    std::fprintf(f_out, "%s\n", line.c_str());
  }
  pending_string_lines.clear();
}

void mir_gen_be() {
  // With IL_SHOULD_BE_WRITTEN_TO_FILE (this configuration), fe_wrapup
  // wrote the IL out and freed it; read it back exactly like
  // c_gen_be's back_end does. skip_il_read means it never left memory.
#if IL_SHOULD_BE_WRITTEN_TO_FILE
  primary_source_file_name = NULL;
  if (!skip_il_read) {
    il_read(f_il_output);
  }
  primary_source_file_name = il_header.primary_source_file->file_name;
#endif
  const char *path = gen_c_file_name;
  if (path == NULL) path = "out.mir";
  f_out = std::fopen(path, "w");
  if (f_out == NULL) {
    std::fprintf(stderr, "glcc mir_gen_be: cannot open output file '%s'\n",
                 path);
    std::exit(3);
  }
  a_scope_ptr scope = il_header.primary_scope;
  // Interfaces to routines not defined in this TU.
  for (a_routine_ptr rout = scope->routines; rout != NULL; rout = rout->next) {
    if (ignore_routine_in_back_end(rout) || rout->superseded_external) continue;
    if (!needed_flag_is_set(&rout->source_corresp)) continue;
    // Implicitly-declared special members (copy/assign ops for
    // __va_list_tag etc.) are never called unless the program uses
    // them; if used, their definition exists and is diagnosed when
    // its body is emitted.
    if (rout->special_kind != (a_special_function_kind)sfk_none) continue;
    if (rout->function_def_number == NULL_function_def_number) {
      emit_extern_routine(rout);
    }
  }
  // String literals referenced from function bodies, hoisted ahead of
  // the bodies so names resolve in one pass.
  for (a_routine_ptr rout = scope->routines; rout != NULL; rout = rout->next) {
    if (ignore_routine_in_back_end(rout) || rout->superseded_external) continue;
    if (!needed_flag_is_set(&rout->source_corresp)) continue;
    if (rout->function_def_number == NULL_function_def_number) continue;
    a_memory_region_number region = mem_region_for_routine(rout);
    if (region == NULL_region_number) continue;
    if (mem_region_table[region] == NULL) read_memory_region(region);
    scan_stmt_strings(scope_for_routine(rout)->assoc_block);
  }
  // String literals reachable from static initializers, so their
  // .global lines precede every relocation that names them.
  for (a_variable_ptr var = scope->variables; var != NULL; var = var->next) {
    if (ignore_variable_in_back_end(var)) continue;
    if (!needed_flag_is_set(&var->source_corresp)) continue;
    if (var->init_kind == (an_init_kind)initk_static) {
      scan_const_strings(var->initializer.constant);
    } else if (var->init_kind == (an_init_kind)initk_dynamic) {
      a_dynamic_init_ptr dip = var->initializer.dynamic;
      if (dip != NULL &&
          dip->kind == (a_dynamic_init_kind)dik_constant) {
        scan_const_strings(dip->variant.constant.ptr);
      }
    }
  }
  flush_pending_strings();
  // Data definitions and extern data declarations.
  for (a_variable_ptr var = scope->variables; var != NULL; var = var->next) {
    if (ignore_variable_in_back_end(var)) continue;
    if (!needed_flag_is_set(&var->source_corresp)) continue;
    emit_variable(var);
  }
  // Function bodies.
  for (a_routine_ptr rout = scope->routines; rout != NULL; rout = rout->next) {
    if (ignore_routine_in_back_end(rout) || rout->superseded_external) continue;
    if (!needed_flag_is_set(&rout->source_corresp)) continue;
    if (rout->function_def_number == NULL_function_def_number) continue;
    a_memory_region_number region = mem_region_for_routine(rout);
    if (region == NULL_region_number) continue;
    if (mem_region_table[region] == NULL) read_memory_region(region);
    emit_func_def(rout, scope_for_routine(rout));
  }
  flush_pending_strings();
  for (const std::string &line : pending_module_lines) {
    std::fprintf(f_out, "%s\n", line.c_str());
  }
  pending_module_lines.clear();
  std::fclose(f_out);
  f_out = NULL;
}

} // namespace

void back_end(void)
// Backend dispatcher: GLCPP_EDG_BACKEND=cgen selects the legacy
// generated-C back end (linked as c_gen_be_back_end); anything else
// takes the MIR path.
{
  const char *mode = std::getenv("GLCPP_EDG_BACKEND");
  if (mode != NULL && std::strcmp(mode, "cgen") == 0) {
    c_gen_be_back_end();
    return;
  }
  mir_gen_be();
}

void mir_gen_be_cleanup(void)
// Companion cleanup hook (parity with c_gen_be_cleanup): closes the
// output file if a compilation terminated before it was flushed.
{
  if (f_out != NULL) {
    std::fclose(f_out);
    f_out = NULL;
  }
}

END_EDG_NAMESPACE

