// Bit-field runs that extend to the end of the class's field list: the
// run scan walks off the list (run_end == NULL), so both locating the
// field within its unit and naming the struct's MIR type must treat a
// NULL run end as "in the run". This shape used to hang the emitter in
// an infinite rescan of the first field.
#include <cstdio>

struct op {
  unsigned int datatype : 3;
};

struct multi {
  unsigned a : 5;
  unsigned b : 11;
  unsigned c : 16; // run a+b+c fills exactly one 32-bit unit at offset 0
  unsigned char d; // non-bit-field after the run
};

int main() {
  op o;
  o.datatype = 4;
  multi m;
  m.a = 31;
  m.b = 2047;
  m.c = 65535;
  m.d = 9;

  printf("%u %u %u %u %u\n", o.datatype, m.a, m.b, m.c, m.d);
  return (o.datatype == 4 && m.a == 31 && m.b == 2047 && m.c == 65535 &&
          m.d == 9)
             ? 0
             : 1;
}
