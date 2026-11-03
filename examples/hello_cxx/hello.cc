#include <iostream>

// Function template
template <typename T>
inline T add(const T& a, const T& b) {
    return a + b;
}

// Variadic template
template <typename T>
inline T sum(T value) {
    return value;
}

template <typename T, typename... Rest>
inline T sum(T first, Rest... rest) {
    return first + sum<T>(rest...);
}

// Class template, with mutable and const reference accessors
template <typename T>
class Box {
    T value;

public:
inline     explicit Box(const T& v) : value(v) {}

    inline T& get() { return value; }
    inline const T& get() const { return value; }
};

// Explicit template specialization
template <typename T>
struct TypeName {
inline     static const char* get() { return "unknown"; }
};

template <>
struct TypeName<int> {
    inline static const char* get() { return "int"; }
};

// References
inline void increment(int& value) {
    ++value;
}

inline const int& identity(const int& value) {
    return value;
}

// Operator overloading and iostream
struct Pair {
    int first;
    int second;

    inline Pair(int a, int b) : first(a), second(b) {}
};

inline std::ostream& operator<<(std::ostream& out, const Pair& p) {
    return out << '(' << p.first << ", " << p.second << ')';
}

// Exception unwinding
struct UnwindProbe {
    bool& unwound;

    inline explicit UnwindProbe(bool& flag) : unwound(flag) {}

    inline ~UnwindProbe() {
        unwound = true;
    }
};

// Inheritance and virtual dispatch
struct Base {
    inline virtual int get() const {
        return 1;
    }

    inline virtual ~Base() {}
};

struct Derived : Base {
    inline int get() const {
        return 9;
    }
};

int main() {
    int value = 7;

    int& alias = value;
    increment(alias);

    const int& view = identity(value);

    Box<int> box(add(view, 10));
    box.get() += 1;

    const Box<int>& const_box = box;

    int packed = sum(1, 2, 3, 4);

    Pair pair(3, 4);

    Derived derived;
    Base& polymorphic = derived;
    int virtual_value = polymorphic.get();

    bool unwound = false;
    int caught = 0;

    // TODO: fix miscompile
    // try {
        // UnwindProbe probe(unwound);
        // throw 42;
    // } catch (const int& error) {
        // caught = error;
    // }

    std::cout << "hello world\n";

    std::cout << "template.add=" << add(4, 5) << '\n';
    std::cout << "template.pack=" << packed << '\n';
    std::cout << "template.box=" << const_box.get()
              << " (" << TypeName<int>::get() << ")\n";

    std::cout << "reference=" << value
              << ", const-ref=" << view << '\n';

    std::cout << "exception=" << caught
              << ", unwound=" << unwound << '\n';

    std::cout << "operator<<=" << pair << '\n';
    std::cout << "virtual=" << virtual_value << '\n';

    return 0;
}
