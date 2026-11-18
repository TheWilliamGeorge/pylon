# Pylon

<img width="256" height="256" alt="Pylon Logo" src="https://github.com/user-attachments/assets/6cd20592-14ac-4385-9fb0-4ae364d621bf" />

Pylon is a dialect of Python with **Java-like syntax** — braced blocks, semicolons,
and explicit static typing — that compiles down to **Cython**, then to native C,
and runs as compiled machine code. It exists to give you the ergonomics of Java
syntax on top of all of Python's batteries, while feeding your type annotations
straight into Cython so the hot paths run at C speed. This means that Pylon can use any Python module, keeping all the flexbility of Python. 

```
.pylon source  →  Pylon AST  →  Cython .pyx  →  C (.c)  →  native .so  →  run
```

## Why

Python is an amazing language. It is simple, readable and easy to learn and use, while still being very powerful. On the other hand, languages like Java feature clear type declarations and tidier syntax using {} and semicolons, which can make code easier and cleaerer to read in some cases. It is also much closer to the syntax of many other languages.
One massive limitation of Python is its speed. C, on the other hand, while being very complex and hard to learn, is blazingly fast. A project named Cython exists to compile Python code into C, allowing great speed increases, especially when variables are concerned. This is because no Python variable types are dynamic, slowing down execution. 
Pylon is the intersection: every typed variable and typed function parameter becomes a C declaration in the generated `.pyx`. So, `int`/`float`/`boolean` arithmetic runs on bare C `long`/`double`/`bint`, while `string`, `list`, `dict`, `bytes`, ... stay as Python objects and keep their full semantics.

## Install

Pylon needs **Python 3**, **Cython**, and a **C compiler** (gcc/cc).

```bash
pip install cython
# gcc is required; on Debian/Ubuntu: sudo apt install build-essential
```

The translator itself is a single file, `pylonc.py`, with no dependencies beyond
the Python standard library.

## Usage

```bash
# Default: translate, compile to a .so, run it.
python3 pylonc.py hello.pylon

# Translate and compile but do NOT run. Saves the .pyx, .c, and .so.
python3 pylonc.py hello.pylon --no-run

# Only emit the Cython .pyx (no compile, no run).
python3 pylonc.py hello.pylon --emit-pyx --show-pyx

# Build a STANDALONE executable (cython --embed + link libpython).
# Useful for shipping a binary, but less portable than the .so default.
python3 pylonc.py hello.pylon --embed

# Pass argv through to the program.
python3 pylonc.py hello.pylon --args one two three
```

Flags:

| Flag            | Effect                                                            |
| --------------- | ----------------------------------------------------------------- |
| `--no-run`      | Translate + compile only; keep `.pyx`, `.c`, and `.so`.           |
| `--emit-pyx`    | Stop after writing the `.pyx`. Implies `--no-run`.                |
| `--show-pyx`    | Print the generated `.pyx` to stdout in addition to saving it.  |
| `--embed`       | Build a standalone executable instead of a `.so`.                 |
| `--keep-dir DIR`| Where to drop intermediate files (default `<stem>_build/`).      |
| `-o NAME`       | Name of the output binary (default: same stem as the source).     |
| `--args ...`    | Trailing args forwarded to the program as `sys.argv`.             |

## The Pylon language

### Types

| Pylon type   | Cython backing   | Notes                                         |
| ------------ | ---------------- | --------------------------------------------- |
| `string`     | `str`            | Python str (NOT `str` or `String` in source)  |
| `boolean`    | `bint`           | C int used as a boolean — fast                |
| `int`        | `long`           | C long                                        |
| `float`      | `double`         | C double                                      |
| `complex`    | Python `complex` | kept as object for full semantics             |
| `bytes`      | `bytes`          |                                               |
| `bytearray`  | `bytearray`      |                                               |
| `list`       | `list`           |                                               |
| `tuple`      | `tuple`          |                                               |
| `dict`       | `dict`           |                                               |
| `set`        | `set`            |                                               |
| `frozenset`  | `frozenset`      |                                               |
| `memoryview` | `memoryview`     |                                               |
| `NoneType`   | (omitted)        | as a return type means "returns nothing"      |
| `object`     | `object`         | untyped escape hatch                          |

### Syntax

Pylon looks like Python wearing a Java T-Shirt - many of the features remain the same, but it has a Java-like style on top:

```python
# Typed module-level variables
string name = "james";
int age = 30;
const float PI = 3.14159265358979;

# Typed function with a return type and typed params.
def string HelloWorld(string nameIn) {
    if not nameIn {
        return;
    } else {
        return (f"Hello, {nameIn}");
    }
}

print(HelloWorld(name));
```

The differences from Python are exactly:

1. **Typed declarations**: `Type name = expr;` and `const Type name = expr;`
2. **Typed function headers**: `def ReturnType name(TypedParam p, ...) { body }`
3. **Braced blocks** for bodies, and **semicolons** to terminate statements.

Everything else — imports, `if`/`elif`/`else`, `while`, `for ... in ...`,
`try`/`except`/`finally`, `with`, `class`, list/dict/set comprehensions,
f-strings, exceptions, `raise`, `assert`, `global`/`nonlocal`, `del`, `yield`,
`lambda` — is ordinary Python syntax. You can even write `import numpy as np;`
and use it directly.

### Classes

Pylon supports `class`. Inside a class body, methods are written Python-style
with an explicit `self`, and the constructor is `__init__`:

```python
class Counter {
    def NoneType __init__(self, int start) {
        self.count = start;
    }
    def NoneType inc(self) {
        self.count = self.count + 1;
    }
    def int value(self) {
        return self.count;
    }
}
```

Inside a class body, typed declarations (`int count;`) become plain attribute
assignments — Cython forbids `cdef` attributes in a regular `class`. If you want
C-level attributes, that's a future `cdef class` extension.

### A larger example

See `examples/features.pylon`, which exercises every supported type, functions,
classes, recursion, try/except/finally, `with`, f-strings, and constants.

## How the type hints reach C

This is the whole point. Take this Pylon:

```python
def int factorial(int n) {
    if n <= 1 { return 1; }
    return n * factorial(n - 1);
}
```

Pylon emits this Cython:

```python
cpdef long factorial(long n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
```

`long` parameters and a `long` return type mean `factorial` runs as native C
arithmetic with no Python object boxing on the hot path. The same function in
plain Python would allocate a `PyLong` for every multiply. Recursive factorial
of 10 is microseconds either way; a tight numeric loop over a million
iterations is where this pays off.

## Limitations and design choices

- **Extension modules don't link libpython.** The default build produces a
  `.so` (or `.dylib`-style `.so` on macOS) CPython extension module, NOT a
  linked-against-libpython binary. This is exactly how `setuptools` and
  `cythonize` build extensions: the Python C-API symbols (`Py_None`,
  `_Py_Dealloc`, …) are resolved at runtime by the Python process that
  imports the `.so`. This is what makes the build portable across Linux,
  macOS framework Python, Homebrew Python, pyenv, uv, etc. — there is no
  libpython path to get wrong.
  - On macOS, the `.so` link uses `-undefined dynamic_lookup` (the standard
    flag for deferring symbol resolution to import time).
  - The standalone `--embed` executable path *does* link libpython, and is
    platform-aware: on macOS it uses `-F<frameworks> -framework Python`
    with an rpath, derived from `LIBDIR`; on Linux it uses
    `-L<libdir> -lpython3.x` with an rpath.
- **Division safety**: Pylon does NOT enable Cython's `cdivision=True` directive.
  That directive would make `a / b` for two C ints perform raw C division,
  which traps (SIGFPE / Illegal instruction) on divide-by-zero instead of
  raising Python's `ZeroDivisionError`. Keeping CPython's division semantics
  costs a few cycles and matches what a Python programmer expects. (If you
  want true integer division of two `int`s, write `//`.)
- **`complex`** is kept as a Python `complex` object rather than C
  `double complex`, so you can assign literals like `1.5 + 2.5j` without
  explicit conversion and keep full Python complex semantics.
- **No Python-style INDENT/DEDENT**: Pylon is brace-driven. Indentation is
  cosmetic.
- **`range` as a declared variable type** is mapped to `object` (you usually
  call `range(...)`, you don't declare a variable of type `range`).
- **Constructor style**: classes use Python's `__init__`, not Java's
  `ClassName(...)`. Java-style constructors are a possible future extension.
- Any imported Python modules must be installed on the system. Pylon does not compile these modules as well as the code (unlike a dedicated Python compiler such as Nuitka), so any lines of code using modules written purely in Python cannot receive the speed benefits of Pylon.
