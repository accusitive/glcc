// RTTI: dynamic_cast down / across a hierarchy and typeid equality
// (names are printed only as 0/1 comparisons to stay stable across
// implementations).
#include <cstdio>
#include <typeinfo>

struct Animal {
  virtual ~Animal() {}
};
struct Dog : Animal {
  int leash;
};
struct Cat : Animal {
  int whiskers;
};

int main() {
  Dog d;
  d.leash = 1;
  Animal *pa = &d;

  Dog *pd = dynamic_cast<Dog *>(pa);
  Cat *pc = dynamic_cast<Cat *>(pa);
  int is_dog = (pd != 0 && pd->leash == 1);
  int is_cat = (pc != 0);

  Animal &ra = d;
  int ref_ok = (&dynamic_cast<Dog &>(ra) == &d);
  int same_type = (typeid(*pa) == typeid(Dog));
  int diff_type = (typeid(Dog) != typeid(Cat));
  int base_never = (typeid(*pa) != typeid(Animal));

  printf("%d %d %d %d %d %d\n", is_dog, is_cat, ref_ok, same_type, diff_type,
         base_never);
  return (is_dog == 1 && is_cat == 0 && ref_ok == 1 && same_type == 1 &&
          diff_type == 1 && base_never == 1)
             ? 0
             : 1;
}
