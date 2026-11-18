#!/usr/bin/env python3
"""
pylonc.py - The Pylon translator.

Pylon is a Python dialect with Java-like syntax (braced blocks, semicolons,
explicit static typing) that compiles down to Cython (.pyx) and then to
native code via Cython + gcc, for speed while keeping Python semantics
where it matters.

Pipeline:
    .pylon source  ->  Pylon AST  ->  Cython (.pyx)  ->  C (.c)  ->  native exe  ->  run

Usage:
    pylonc <file.pylon>                  # translate, compile, run
    pylonc <file.pylon> --no-run        # translate + compile only, save .pyx/.c
    pylonc <file.pylon> --emit-pyx      # only emit .pyx (no compile)
    pylonc <file.pylon> -o <out>        # name the output executable
    pylonc <file.pylon> --keep-dir DIR  # where to drop intermediate files
    pylonc <file.pylon> --no-embed      # build a .so (importable) instead of exe
    pylonc --help
"""

from __future__ import annotations

import sys
import os
import re
import shutil
import subprocess
import argparse
from dataclasses import dataclass, field
from typing import Optional, Union

# ---------------------------------------------------------------------------
# Type mapping: Pylon type name  ->  Cython declaration type
# ---------------------------------------------------------------------------
#
# Primitive numeric / bool types use C value types (fast, the whole point of
# feeding Pylon's type hints into Cython). Container and string types stay as
# Python objects so semantics match CPython. NoneType maps to void for return
# types (rare as a variable type).
#
# Pylon name      Cython type      Notes
# -----------     -----------      ----------------------------------------
# string          str              Python str object
# boolean         bint             C int used as a boolean (0/1)
# int             long             C long  (fast; note: no big-int)
# float           double           C double
# complex         doublecomplex    C double complex
# bytes           bytes            Python bytes object
# bytearray       bytearray        Python bytearray object
# list            list             Python list object
# tuple           tuple            Python tuple object
# dict            dict             Python dict object
# set             set              Python set object
# frozenset       frozenset        Python frozenset object
# range           range            Python range object (rarely a var type)
# memoryview      memoryview       Python memoryview object
# NoneType        void             (return type) / object (variable, rare)

PRIMITIVE_C_TYPES = {
    "int":       "long",
    "float":     "double",
    "boolean":   "bint",       # C int used as a boolean (0/1) - fast path
    # Pylon `complex` -> Python complex object (see OBJECT_TYPES). We
    # deliberately do NOT use Cython's C `double complex` here because (a)
    # it requires a libc.complex cimport that isn't always available, and
    # (b) assigning Python complex literals (1.5 + 2.5j) to a C complex
    # needs explicit conversion that would surprise the user. Keeping it as
    # a Python object preserves full Python complex semantics.
}

OBJECT_TYPES = {
    "string":     "str",
    "str":        "str",          # accept Python spelling too, for familiarity
    "bool":       "object",       # accept Python spelling -> object bint? no: keep bint
    "complex":    "object",       # Python complex object (see note above)
    "bytes":      "bytes",
    "bytearray":  "bytearray",
    "list":       "list",
    "tuple":      "tuple",
    "dict":       "dict",
    "set":        "set",
    "frozenset":  "frozenset",
    "range":      "object",       # `range` is usually a call, not a declared type
    "memoryview": "memoryview",
    "NoneType":   "object",
    "object":     "object",
}

# Pylon type name -> Cython return-type spelling (used for cpdef headers).
# NoneType as a return type means "returns nothing" -> emit no return type.
RETURN_TYPE_MAP = {
    "int":       "long",
    "float":     "double",
    "complex":   "object",
    "boolean":   "bint",
    "string":    "str",
    "str":       "str",
    "bool":      "bint",
    "bytes":     "bytes",
    "bytearray": "bytearray",
    "list":      "list",
    "tuple":     "tuple",
    "dict":      "dict",
    "set":       "set",
    "frozenset": "frozenset",
    "memoryview":"memoryview",
    "range":     "object",
    "NoneType":  None,           # signal: omit return type
    "object":    "object",
}

VALID_TYPE_NAMES = (
    set(PRIMITIVE_C_TYPES)
    | set(OBJECT_TYPES)
    | {"NoneType", "object"}
)


def pylon_type_to_cython(name: str) -> str:
    """Map a Pylon type name to a Cython C/object declaration type."""
    if name in PRIMITIVE_C_TYPES:
        return PRIMITIVE_C_TYPES[name]
    if name in OBJECT_TYPES:
        return OBJECT_TYPES[name]
    raise PylonError(f"Unknown Pylon type: {name!r}")


def pylon_type_to_return(name: str) -> Optional[str]:
    """Map a Pylon type name to a Cython return-type annotation (or None)."""
    if name not in RETURN_TYPE_MAP:
        raise PylonError(f"Unknown Pylon type used as return type: {name!r}")
    return RETURN_TYPE_MAP[name]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class PylonError(Exception):
    """Any error raised during Pylon lexing, parsing, or emission."""


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

# Token kinds. Single chars use the literal character (e.g. "{", ";", "=").
# Multi-char / named tokens use these sentinels.
TT_NAME      = "NAME"
TT_NUMBER    = "NUMBER"
TT_STRING    = "STRING"      # includes f-string prefix
TT_FSTRING   = "FSTRING"     # tokenised f-string body (raw, incl. quotes)
TT_OP        = "OP"          # multi-char operators like ==, !=, <=, ->, //
TT_NEWLINE   = "NEWLINE"
TT_EOF       = "EOF"

# Multi-character operators, longest first so the matcher is greedy.
_MULTI_OPS = sorted([
    "**=", "//=", ">>=", "<<=", ":=", "==", "!=", "<=", ">=",
    "**", "//", ">>", "<<", "+=", "-=", "*=", "/=", "%=", "&=", "|=",
    "^=", "->", "and", "or", "not", "in", "is",  # word ops handled below
], key=len, reverse=True)

# Word operators that read like identifiers but are operators.
_WORD_OPS = {"and", "or", "not", "in", "is"}

# These keywords start statements/declarations and are NOT type names.
_KEYWORDS = {
    "def", "if", "elif", "else", "while", "for", "return", "break",
    "continue", "pass", "import", "from", "as", "class", "global",
    "nonlocal", "del", "with", "try", "except", "finally", "raise",
    "assert", "lambda", "yield", "True", "False", "None",
}


@dataclass
class Token:
    kind: str
    value: str
    line: int
    col: int

    def __repr__(self) -> str:
        return f"Token({self.kind!r}, {self.value!r}, line={self.line})"


def tokenize(src: str) -> list[Token]:
    """Convert Pylon source text into a flat token list.

    Pylon uses # line comments, Python-style string literals (incl. f-strings,
    raw strings, triple-quoted, and adjacent string concatenation is NOT
    supported by the lexer - it's left to the parser to merge if needed)."""
    tokens: list[Token] = []
    i = 0
    line = 1
    col = 1
    n = len(src)

    def at(j: int) -> str:
        return src[j] if j < n else ""

    while i < n:
        c = src[i]

        # Newlines. Pylon statements end with ; but we still track newlines
        # so that we can keep source line numbers and (later) allow
        # continuation. A newline inside a { } block is whitespace; the
        # parser is brace-driven, so we mostly drop NEWLINE tokens but keep
        # one per logical line for error messages. Simplest: drop them and
        # rely on ; and { } for structure. We still bump line/col.
        if c == "\n":
            i += 1
            line += 1
            col = 1
            continue
        if c == "\r":
            i += 1
            continue
        # Whitespace
        if c in " \t":
            i += 1
            col += 1
            continue
        # Line continuation: backslash + newline
        if c == "\\" and at(i + 1) == "\n":
            i += 2
            line += 1
            col = 1
            continue
        # Comments
        if c == "#":
            while i < n and src[i] != "\n":
                i += 1
            continue

        # Strings (incl. f/r/b prefixes and triple-quoted)
        if c in "'\"" or (c in "fFrRbBuU" and _peek_string_prefix(src, i)):
            tok, i, col, line = _lex_string(src, i, line, col)
            tokens.append(tok)
            continue

        # Numbers
        if c.isdigit() or (c == "." and at(i + 1).isdigit()):
            start = i
            # Hex / oct / bin
            if c == "0" and at(i + 1) in "xXoObB":
                i += 2
                while i < n and (src[i].isalnum() or src[i] == "_"):
                    i += 1
            else:
                while i < n and (src[i].isdigit() or src[i] == "_"):
                    i += 1
                if i < n and src[i] == ".":
                    i += 1
                    while i < n and (src[i].isdigit() or src[i] == "_"):
                        i += 1
                if i < n and src[i] in "eE":
                    i += 1
                    if i < n and src[i] in "+-":
                        i += 1
                    while i < n and (src[i].isdigit() or src[i] == "_"):
                        i += 1
                if i < n and src[i] in "jJ":
                    i += 1
            tokens.append(Token(TT_NUMBER, src[start:i], line, col))
            col += i - start
            continue

        # Identifiers / keywords / word-operators / type names
        if c.isalpha() or c == "_":
            start = i
            while i < n and (src[i].isalnum() or src[i] == "_"):
                i += 1
            word = src[start:i]
            if word in _WORD_OPS:
                tokens.append(Token(TT_OP, word, line, col))
            else:
                tokens.append(Token(TT_NAME, word, line, col))
            col += i - start
            continue

        # Multi-character operators
        matched = False
        for op in _MULTI_OPS:
            if src.startswith(op, i):
                tokens.append(Token(TT_OP, op, line, col))
                i += len(op)
                col += len(op)
                matched = True
                break
        if matched:
            continue

        # Single-character punctuation/operators. We treat { } ; and ( ) as
        # structural; everything else is a single-char OP. '.' is needed for
        # attribute access (out.append) and for float literals like `.5`.
        if c in "{}[]();,.:":
            tokens.append(Token(c, c, line, col))
            i += 1
            col += 1
            continue
        if c in "+-*/%<>=&|^~@":
            tokens.append(Token(TT_OP, c, line, col))
            i += 1
            col += 1
            continue

        raise PylonError(f"Unexpected character {c!r} at line {line}, col {col}")

    tokens.append(Token(TT_EOF, "", line, col))
    return tokens


def _peek_string_prefix(src: str, i: int) -> bool:
    """Return True if a string prefix (f/r/b/u and combinations) at i is
    immediately followed by a quote char."""
    j = i
    while j < len(src) and src[j] in "fFrRbBuU":
        j += 1
    return j < len(src) and src[j] in "'\""


def _lex_string(src: str, i: int, line: int, col: int):
    """Lex a string (possibly prefixed with f/r/b) starting at index i.
    Returns (Token, new_i, new_col, new_line)."""
    n = len(src)
    start = i
    # Prefix
    prefix_start = i
    while i < n and src[i] in "fFrRbBuU":
        i += 1
    prefix = src[prefix_start:i].lower()
    if i >= n or src[i] not in "'\"":
        raise PylonError(f"Malformed string literal at line {line}")
    quote = src[i]
    # Triple-quoted?
    if src.startswith(quote * 3, i):
        end_seq = quote * 3
        i += 3
        body_start = i
        while i < n and not src.startswith(end_seq, i):
            if src[i] == "\n":
                line += 1
                col = 1
            i += 1
        if i >= n:
            raise PylonError(f"Unterminated triple-quoted string at line {line}")
        i += 3
    else:
        i += 1
        body_start = i
        while i < n and src[i] != quote:
            if src[i] == "\\" and i + 1 < n:
                i += 2
                continue
            if src[i] == "\n":
                raise PylonError(
                    f"Unterminated string literal at line {line} "
                    f"(newline in single-quoted string)"
                )
            i += 1
        if i >= n:
            raise PylonError(f"Unterminated string literal at line {line}")
        i += 1  # closing quote
    raw = src[start:i]
    # f-strings pass through verbatim; Cython/Python parses them.
    kind = TT_FSTRING if "f" in prefix else TT_STRING
    return Token(kind, raw, line, col), i, col + (i - start), line


# ---------------------------------------------------------------------------
# AST nodes
# ---------------------------------------------------------------------------
#
# Pylon's surface syntax differs from Python only in three ways:
#   1. Typed variable declarations:    int x = 5;
#   2. Typed function declarations:    def int square(int n) { ... }
#   3. Braced blocks + semicolons instead of INDENT/DEDENT + NEWLINE.
#
# Everything else is ordinary Python: assignment, control flow keywords,
# expressions. So the AST is a small handful of statement nodes plus an
# expression tree that we keep as raw token slices and emit verbatim with
# only minor rewriting (semicolons gone, braces -> newlines/indent).

@dataclass
class Node:
    pass


@dataclass
class Decl(Node):
    """A typed variable declaration:   Type name = expr ;   (init optional)"""
    type_name: str
    name: str
    init: Optional[list[Token]] = None      # raw expr tokens
    line: int = 0


@dataclass
class ConstDecl(Node):
    """A typed constant declaration:   const Type name = expr ;"""
    type_name: str
    name: str
    init: list[Token]
    line: int = 0


@dataclass
class FuncDef(Node):
    """def ReturnType name ( typed-params ) { body }"""
    return_type: Optional[str]            # None = no return type annotation
    name: str
    params: list[tuple[str, str]]          # [(type_name, param_name), ...]
    body: list[Node]                       # statements
    line: int = 0


@dataclass
class ClassDef(Node):
    """class Name(Base):  { body }   - bases optional"""
    name: str
    bases: list[str]
    body: list[Node]
    line: int = 0


@dataclass
class IfStmt(Node):
    cond: list[Token]
    then_body: list[Node]
    elif_clauses: list[tuple[list[Token], list[Node]]] = field(default_factory=list)
    else_body: Optional[list[Node]] = None
    line: int = 0


@dataclass
class WhileStmt(Node):
    cond: list[Token]
    body: list[Node]
    line: int = 0


@dataclass
class ForStmt(Node):
    """for x in expr { body }   OR   for i = 0; i < n; i++ { body }
    We support the Python `for x in iter` form. The C-style form is a
    possible future extension; the user's example uses Python-style for."""
    target: list[Token]       # name or tuple, raw tokens
    iter_expr: list[Token]
    body: list[Node]
    line: int = 0


@dataclass
class ReturnStmt(Node):
    value: Optional[list[Token]] = None
    line: int = 0


@dataclass
class BreakStmt(Node):
    line: int = 0


@dataclass
class ContinueStmt(Node):
    line: int = 0


@dataclass
class PassStmt(Node):
    line: int = 0


@dataclass
class ImportStmt(Node):
    """import a [as b];  or  from a import b [as c], d"""
    from_module: Optional[str]
    names: list[tuple[str, Optional[str]]]   # [(name, alias)]
    line: int = 0


@dataclass
class RaiseStmt(Node):
    exc: Optional[list[Token]] = None
    line: int = 0


@dataclass
class TryStmt(Node):
    """try { ... } except Exc [as e] { ... } ... finally { ... }"""
    body: list[Node]
    handlers: list["ExceptHandler"]
    else_body: Optional[list[Node]] = None
    finally_body: Optional[list[Node]] = None
    line: int = 0


@dataclass
class ExceptHandler(Node):
    exc_type: Optional[list[Token]] = None
    as_name: Optional[str] = None
    body: list[Node] = field(default_factory=list)
    line: int = 0


@dataclass
class WithStmt(Node):
    items: list[tuple[list[Token], Optional[str]]]   # [(ctx_expr, as_name)]
    body: list[Node]
    line: int = 0


@dataclass
class DelStmt(Node):
    target: list[Token]
    line: int = 0


@dataclass
class GlobalStmt(Node):
    names: list[str]
    line: int = 0


@dataclass
class NonlocalStmt(Node):
    names: list[str]
    line: int = 0


@dataclass
class AssertStmt(Node):
    test: list[Token]
    msg: Optional[list[Token]] = None
    line: int = 0


@dataclass
class ExprStmt(Node):
    """A bare expression or assignment, carried as raw tokens.
    Semicolons are stripped by the parser. Multi-target assignments
    (a = b = c = 0) and augmented assignments (x += 1) are all carried
    here as a token list; the emitter just joins them with spaces."""
    tokens: list[Token]
    line: int = 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class Parser:
    def __init__(self, tokens: list[Token]):
        self.toks = tokens
        self.i = 0

    # -- token cursor helpers ------------------------------------------------
    def peek(self, off: int = 0) -> Token:
        j = self.i + off
        if j >= len(self.toks):
            return self.toks[-1]
        return self.toks[j]

    def next(self) -> Token:
        t = self.toks[self.i]
        if self.i < len(self.toks) - 1:
            self.i += 1
        return t

    def at_end(self) -> bool:
        return self.peek().kind == TT_EOF

    def at_word(self, word: str) -> bool:
        """True if the next token is a NAME or word-OP matching `word`.
        Needed because 'as', 'in', 'is', 'and', 'or', 'not' are lexed as
        TT_OP, while keywords like 'else', 'except' are TT_NAME."""
        t = self.peek()
        return t.kind in (TT_NAME, TT_OP) and t.value == word

    def expect(self, kind: str, value: Optional[str] = None) -> Token:
        t = self.peek()
        if t.kind != kind or (value is not None and t.value != value):
            exp = f"{kind}({value!r})" if value else kind
            raise PylonError(
                f"Expected {exp} but got {t.kind}({t.value!r}) at line {t.line}"
            )
        return self.next()

    def accept(self, kind: str, value: Optional[str] = None) -> Optional[Token]:
        t = self.peek()
        if t.kind == kind and (value is None or t.value == value):
            return self.next()
        return None

    # -- top level ----------------------------------------------------------
    def parse_program(self) -> list[Node]:
        stmts: list[Node] = []
        # Allow leading import / module-level statements / declarations.
        while not self.at_end():
            stmts.append(self.parse_statement())
        return stmts

    # -- statement dispatch --------------------------------------------------
    def parse_statement(self) -> Node:
        t = self.peek()
        if t.kind == TT_NAME:
            kw = t.value
            if kw == "def":
                return self.parse_func()
            if kw == "class":
                return self.parse_class()
            if kw == "if":
                return self.parse_if()
            if kw == "while":
                return self.parse_while()
            if kw == "for":
                return self.parse_for()
            if kw == "return":
                return self.parse_return()
            if kw == "break":
                self.next()
                self.expect(";")
                return BreakStmt(line=t.line)
            if kw == "continue":
                self.next()
                self.expect(";")
                return ContinueStmt(line=t.line)
            if kw == "pass":
                self.next()
                self.expect(";")
                return PassStmt(line=t.line)
            if kw == "import":
                return self.parse_import()
            if kw == "from":
                return self.parse_from_import()
            if kw == "raise":
                return self.parse_raise()
            if kw == "try":
                return self.parse_try()
            if kw == "with":
                return self.parse_with()
            if kw == "del":
                return self.parse_del()
            if kw == "global":
                return self.parse_global()
            if kw == "nonlocal":
                return self.parse_nonlocal()
            if kw == "assert":
                return self.parse_assert()
            if kw == "const":
                return self.parse_const()
            if kw in VALID_TYPE_NAMES:
                # typed variable declaration:  Type name [= expr] ;
                return self.parse_decl()
            # otherwise: a bare expression / assignment statement
        return self.parse_expr_statement()

    # -- typed declaration:  Type name [= expr] ;  --------------------------
    def parse_decl(self) -> Decl:
        type_tok = self.next()               # the type name
        name_tok = self.expect(TT_NAME)
        init = None
        if self.accept(TT_OP, "="):
            init = self.collect_expr_until_semicolon()
        self.expect(";")
        if name_tok.value not in ("_",) and name_tok.value in _KEYWORDS:
            raise PylonError(
                f"Cannot use keyword {name_tok.value!r} as variable name "
                f"(line {name_tok.line})"
            )
        return Decl(type_name=type_tok.value, name=name_tok.value,
                    init=init, line=type_tok.line)

    # -- const declaration:  const Type name = expr ;  ---------------------
    def parse_const(self) -> ConstDecl:
        kw = self.expect(TT_NAME, "const")
        type_tok = self.expect(TT_NAME)
        if type_tok.value not in VALID_TYPE_NAMES:
            raise PylonError(
                f"Unknown type {type_tok.value!r} in const declaration "
                f"at line {type_tok.line}"
            )
        name_tok = self.expect(TT_NAME)
        self.expect(TT_OP, "=")
        init = self.collect_expr_until_semicolon()
        self.expect(";")
        return ConstDecl(type_name=type_tok.value, name=name_tok.value,
                         init=init, line=kw.line)

    # -- def ReturnType name ( typed params ) { body }  --------------------
    def parse_func(self) -> FuncDef:
        kw = self.expect(TT_NAME, "def")
        # Return type: an optional type name preceding the function name.
        # We only consume it if the NEXT token (the one after the candidate
        # type) is a NAME that's not a type - i.e. "Type funcName". This
        # avoids eating the function name itself when there's no return type.
        return_type: Optional[str] = None
        if (self.peek().kind == TT_NAME
                and self.peek().value in VALID_TYPE_NAMES
                and self.peek(1).kind == TT_NAME
                and self.peek(1).value not in VALID_TYPE_NAMES):
            return_type = self.next().value
        name_tok = self.expect(TT_NAME)
        self.expect("(")
        params: list[tuple[str, str]] = []
        if not self.accept(")"):
            while True:
                # A parameter is either "Type name" or a bare "name"
                # (untyped - used for `self`, `cls`, or any Python-object arg).
                first = self.expect(TT_NAME)
                if first.value in VALID_TYPE_NAMES and self.peek().kind == TT_NAME \
                        and self.peek().value not in _KEYWORDS:
                    # Type name followed by param name
                    ptype = first.value
                    pname = self.expect(TT_NAME).value
                else:
                    # Bare name (e.g. self, cls, or untyped arg)
                    ptype = "object"
                    pname = first.value
                params.append((ptype, pname))
                if self.accept(","):
                    # trailing comma ok
                    if self.peek().kind == ")":
                        break
                    continue
                break
            self.expect(")")
        body = self.parse_block()
        return FuncDef(return_type=return_type, name=name_tok.value,
                       params=params, body=body, line=kw.line)

    # -- class Name(Base, ...): { body }  -----------------------------------
    def parse_class(self) -> ClassDef:
        kw = self.expect(TT_NAME, "class")
        name_tok = self.expect(TT_NAME)
        bases: list[str] = []
        if self.accept("("):
            if not self.accept(")"):
                while True:
                    b = self.expect(TT_NAME)
                    bases.append(b.value)
                    if self.accept(","):
                        if self.peek().kind == ")":
                            break
                        continue
                    break
                self.expect(")")
        body = self.parse_block()
        return ClassDef(name=name_tok.value, bases=bases, body=body, line=kw.line)

    # -- if / elif / else  --------------------------------------------------
    def parse_if(self) -> IfStmt:
        kw = self.expect(TT_NAME, "if")
        cond = self.collect_expr_until("{")
        self.expect("{")
        then_body = self.parse_block_body()
        elif_clauses: list[tuple[list[Token], list[Node]]] = []
        else_body: Optional[list[Node]] = None
        while True:
            # peek for `else` or `elif` (but only when not nested; the block
            # parser already consumed its own closing brace).
            if self.peek().kind == TT_NAME and self.peek().value == "elif":
                self.next()
                ec = self.collect_expr_until("{")
                self.expect("{")
                eb = self.parse_block_body()
                elif_clauses.append((ec, eb))
            elif self.peek().kind == TT_NAME and self.peek().value == "else":
                self.next()
                self.expect("{")
                else_body = self.parse_block_body()
                break
            else:
                break
        return IfStmt(cond=cond, then_body=then_body, elif_clauses=elif_clauses,
                      else_body=else_body, line=kw.line)

    # -- while  -------------------------------------------------------------
    def parse_while(self) -> WhileStmt:
        kw = self.expect(TT_NAME, "while")
        cond = self.collect_expr_until("{")
        self.expect("{")
        body = self.parse_block_body()
        return WhileStmt(cond=cond, body=body, line=kw.line)

    # -- for x in iter { ... }  ---------------------------------------------
    def parse_for(self) -> ForStmt:
        kw = self.expect(TT_NAME, "for")
        # target: a name, or a parenthesised tuple of names
        target = self.collect_expr_until_keyword("in")
        in_tok = self.expect(TT_OP, "in")
        iter_expr = self.collect_expr_until("{")
        self.expect("{")
        body = self.parse_block_body()
        return ForStmt(target=target, iter_expr=iter_expr, body=body, line=kw.line)

    # -- return [expr] ;  ---------------------------------------------------
    def parse_return(self) -> ReturnStmt:
        kw = self.expect(TT_NAME, "return")
        if self.accept(";"):
            return ReturnStmt(value=None, line=kw.line)
        value = self.collect_expr_until_semicolon()
        self.expect(";")
        return ReturnStmt(value=value, line=kw.line)

    # -- import a [as b] ;  -------------------------------------------------
    def parse_import(self) -> ImportStmt:
        kw = self.expect(TT_NAME, "import")
        names: list[tuple[str, Optional[str]]] = []
        # dotted module name; we keep dotted as one string per import target.
        while True:
            mod = self.expect(TT_NAME).value
            while self.accept(TT_OP, "."):
                mod += "." + self.expect(TT_NAME).value
            alias = None
            if self.at_word("as"):
                self.next()
                alias = self.expect(TT_NAME).value
            names.append((mod, alias))
            if self.accept(","):
                continue
            break
        self.expect(";")
        return ImportStmt(from_module=None, names=names, line=kw.line)

    # -- from a import b [as c], d ;  ---------------------------------------
    def parse_from_import(self) -> ImportStmt:
        kw = self.expect(TT_NAME, "from")
        mod = self.expect(TT_NAME).value
        while self.accept(TT_OP, "."):
            mod += "." + self.expect(TT_NAME).value
        imp = self.expect(TT_NAME, "import")
        names: list[tuple[str, Optional[str]]] = []
        if self.peek().kind == TT_OP and self.peek().value == "*":
            self.next()
            names.append(("*", None))
        else:
            while True:
                n = self.expect(TT_NAME).value
                alias = None
                if self.at_word("as"):
                    self.next()
                    alias = self.expect(TT_NAME).value
                names.append((n, alias))
                if self.accept(","):
                    continue
                break
        self.expect(";")
        return ImportStmt(from_module=mod, names=names, line=kw.line)

    # -- raise [expr] ;  ----------------------------------------------------
    def parse_raise(self) -> RaiseStmt:
        kw = self.expect(TT_NAME, "raise")
        if self.accept(";"):
            return RaiseStmt(exc=None, line=kw.line)
        exc = self.collect_expr_until_semicolon()
        self.expect(";")
        return RaiseStmt(exc=exc, line=kw.line)

    # -- try { } except ... { } [finally { }]  ------------------------------
    def parse_try(self) -> TryStmt:
        kw = self.expect(TT_NAME, "try")
        self.expect("{")
        body = self.parse_block_body()
        handlers: list[ExceptHandler] = []
        else_body: Optional[list[Node]] = None
        finally_body: Optional[list[Node]] = None
        while self.peek().kind == TT_NAME and self.peek().value == "except":
            self.next()
            eh = ExceptHandler(line=self.peek().line)
            # optional exception type expression (dotted name or tuple)
            if self.peek().kind != "{":
                eh.exc_type = self.collect_expr_until_keyword("as", "{")
                if self.at_word("as"):
                    self.next()
                    eh.as_name = self.expect(TT_NAME).value
            self.expect("{")
            eh.body = self.parse_block_body()
            handlers.append(eh)
        if self.peek().kind == TT_NAME and self.peek().value == "else":
            self.next()
            self.expect("{")
            else_body = self.parse_block_body()
        if self.peek().kind == TT_NAME and self.peek().value == "finally":
            self.next()
            self.expect("{")
            finally_body = self.parse_block_body()
        return TryStmt(body=body, handlers=handlers, else_body=else_body,
                       finally_body=finally_body, line=kw.line)

    # -- with ctx [as v] , ... { body }  ------------------------------------
    def parse_with(self) -> WithStmt:
        kw = self.expect(TT_NAME, "with")
        items: list[tuple[list[Token], Optional[str]]] = []
        while True:
            ctx = self.collect_expr_until_keyword("as", "{")
            as_name = None
            if self.at_word("as"):
                self.next()
                as_name = self.expect(TT_NAME).value
            items.append((ctx, as_name))
            if self.accept(","):
                continue
            break
        self.expect("{")
        body = self.parse_block_body()
        return WithStmt(items=items, body=body, line=kw.line)

    # -- del target ;  ------------------------------------------------------
    def parse_del(self) -> DelStmt:
        kw = self.expect(TT_NAME, "del")
        target = self.collect_expr_until_semicolon()
        self.expect(";")
        return DelStmt(target=target, line=kw.line)

    # -- global a, b ;  -----------------------------------------------------
    def parse_global(self) -> GlobalStmt:
        kw = self.expect(TT_NAME, "global")
        names = [self.expect(TT_NAME).value]
        while self.accept(","):
            names.append(self.expect(TT_NAME).value)
        self.expect(";")
        return GlobalStmt(names=names, line=kw.line)

    def parse_nonlocal(self) -> NonlocalStmt:
        kw = self.expect(TT_NAME, "nonlocal")
        names = [self.expect(TT_NAME).value]
        while self.accept(","):
            names.append(self.expect(TT_NAME).value)
        self.expect(";")
        return NonlocalStmt(names=names, line=kw.line)

    # -- assert expr [, msg] ;  ---------------------------------------------
    def parse_assert(self) -> AssertStmt:
        kw = self.expect(TT_NAME, "assert")
        test = self.collect_expr_until_comma_or_semicolon()
        msg = None
        if self.accept(","):
            msg = self.collect_expr_until_semicolon()
        self.expect(";")
        return AssertStmt(test=test, msg=msg, line=kw.line)

    # -- bare expression / assignment statement  -----------------------------
    def parse_expr_statement(self) -> ExprStmt:
        line = self.peek().line
        toks = self.collect_expr_until_semicolon()
        self.expect(";")
        if not toks:
            raise PylonError(f"Empty statement at line {line}")
        return ExprStmt(tokens=toks, line=line)

    # -- block parsing -------------------------------------------------------
    def parse_block(self) -> list[Node]:
        """Expect a `{`, parse statements until `}`."""
        self.expect("{")
        return self.parse_block_body()

    def parse_block_body(self) -> list[Node]:
        """Parse statements until the matching closing `}` (already consumed
        by the caller via expect('}'))."""
        stmts: list[Node] = []
        while True:
            t = self.peek()
            if t.kind == "}":
                self.next()
                break
            if t.kind == TT_EOF:
                raise PylonError(
                    f"Unexpected end of file inside block (missing '}}') "
                    f"started around line {t.line}"
                )
            stmts.append(self.parse_statement())
        return stmts

    # -- expression-collection helpers --------------------------------------
    #
    # Pylon's expressions are exactly Python expressions, so we don't write
    # a full expression parser: we collect raw tokens until we hit a stopper
    # that this language uses as a structural delimiter at the statement
    # level. Stoppers: ; { }  and certain keywords (in, as, else, elif,
    # except, finally) when they appear at paren-depth 0.
    #
    def collect_expr_until(self, stop_char: str) -> list[Token]:
        return self._collect(stop_chars={stop_char}, stop_words=set())

    def collect_expr_until_semicolon(self) -> list[Token]:
        return self._collect(stop_chars={";"}, stop_words=set())

    def collect_expr_until_comma_or_semicolon(self) -> list[Token]:
        return self._collect(stop_chars={",", ";"}, stop_words=set())

    def collect_expr_until_keyword(self, *words: str) -> list[Token]:
        ws = set(words)
        return self._collect(stop_chars={"{"}, stop_words=ws)

    def _collect(self, stop_chars: set[str], stop_words: set[str]) -> list[Token]:
        """Collect expression tokens until a structural stopper.

        Stop conditions (checked ONLY at paren/bracket/brace depth 0):
          - a char in `stop_chars` (e.g. ';', '{')
          - a '{' or '}'  (block delimiters are never part of an expression
            at depth 0 - they're set/dict literals only when nested inside
            another brace, which means depth > 0)
          - a NAME in `stop_words` (e.g. 'in', 'as')

        Depth tracking covers (), [], AND {} so that set/dict literals and
        nested comprehension clauses inside the expression are collected as
        a unit."""
        out: list[Token] = []
        depth = 0
        while True:
            t = self.peek()
            if t.kind == TT_EOF:
                raise PylonError(
                    f"Unexpected EOF while collecting expression at line {t.line}"
                )
            if depth == 0:
                if t.kind in stop_chars:
                    break
                # Word operators like 'in', 'as', 'is', 'and', 'or', 'not'
                # are lexed as TT_OP, not TT_NAME, so check both kinds.
                if t.kind in (TT_NAME, TT_OP) and t.value in stop_words:
                    break
                # '{' / '}' at depth 0:
                #  - if '{' is an explicit stop char, stop (caller wants it).
                #  - otherwise '{' as the FIRST token of the collected
                #    expression is a set/dict literal - start a nested group.
                #  - any other '{' or '}' at depth 0 ends the expression
                #    (it's a block delimiter, which the caller will handle).
                if t.kind == "{" and "{" not in stop_chars and not out:
                    pass  # treat first-token '{' as a literal start
                elif t.kind == "{" or t.kind == "}":
                    break
            # Track depth across all bracket kinds.
            if t.kind in "([{":
                depth += 1
            elif t.kind in ")]}":
                if depth == 0:
                    raise PylonError(
                        f"Unbalanced '{t.value}' at line {t.line}"
                    )
                depth -= 1
            out.append(self.next())
        return out


# ---------------------------------------------------------------------------
# Cython emitter
# ---------------------------------------------------------------------------
#
# Walks the Pylon AST and produces a .pyx file. The emitter uses a simple
# indentation stack and joins expression tokens back into source text via
# `render_tokens`.

def render_tokens(toks: list[Token]) -> str:
    """Turn a flat token list back into a source string with sensible
    spacing. Rules:
      - no space before , ) ] ; : .
      - no space after ( [ .
      - single space around binary operators and = (but not <= >= == etc.
        get no extra padding)
      - adjacent NAME/NUMBER/STRING get a space
    For our purposes, a near-Python rendering is enough: Cython parses it
    the same way Python would. We bias toward always inserting a space
    unless two tokens are clearly meant to be adjacent (e.g. '(' then x).
    """
    if not toks:
        return ""
    parts: list[str] = []
    for idx, t in enumerate(toks):
        s = t.value
        prev = toks[idx - 1] if idx > 0 else None
        if idx == 0:
            parts.append(s)
            continue
        # Decide whether to insert a space before this token.
        join = " "
        if prev is not None:
            pv = prev.value
            # No space after an opening bracket or dot
            if pv in "([":
                join = ""
            elif pv == ".":
                join = ""
            # No space before a closing bracket, comma, semicolon, colon, dot
            if s in ")]":
                join = ""
            if s.startswith(",") or s.startswith(";") or s.startswith(":") or s.startswith("."):
                join = ""
            # Adjacency of function-call: name( -> no space
            if (prev.kind in (TT_NAME, TT_NUMBER, TT_STRING, TT_FSTRING)
                    and s in "(["):
                join = ""
            # Two adjacent names/numbers need a space (avoid `a b`)
            if prev.kind in (TT_NAME, TT_NUMBER) and t.kind in (TT_NAME, TT_NUMBER):
                join = " "
            # Word-operator adjacency: keep space around word ops
            if prev.kind == TT_OP and prev.value in {"and", "or", "not", "in", "is"}:
                join = " "
            if t.kind == TT_OP and t.value in {"and", "or", "not", "in", "is"}:
                join = " "
            # Don't double-space; collapse any leading space if join == ""
            parts[-1] = parts[-1]  # no-op; just for clarity
        parts.append(join)
        parts.append(s)
    return "".join(parts)


class Emitter:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.indent = 0
        # Track declared module-level typed constants so we can emit them as
        # cdef globals where it's safe.
        self._in_func = 0
        # When > 0, we're inside a `class` body. Inside a class, typed
        # declarations become plain attribute assignments (Cython forbids
        # `cdef` attrs in a regular `class`, and `cdef class` has its own
        # __cinit__ rules we don't want to require). Methods also become
        # plain `def` (cpdef is only legal in cdef classes).
        self._in_class = 0

    # -- low-level ----------------------------------------------------------
    def emit(self, line: str = "") -> None:
        if line:
            self.lines.append("    " * self.indent + line)
        else:
            self.lines.append("")

    def push(self) -> None:
        self.indent += 1

    def pop(self) -> None:
        self.indent = max(0, self.indent - 1)

    def render(self) -> str:
        return "\n".join(self.lines) + "\n"

    # -- module top ---------------------------------------------------------
    def emit_module(self, prog: list[Node]) -> str:
        # Cython file header. The directives trade Python safety checks for
        # speed, which is the whole point of feeding Pylon's type hints into
        # Cython. We deliberately do NOT enable cdivision=True: it would
        # make `a / b` for C ints perform raw C division, which traps
        # (SIGFPE / Illegal instruction) on divide-by-zero instead of
        # raising Python's ZeroDivisionError. Keeping CPython's division
        # semantics is worth the tiny overhead and matches what a Python
        # programmer expects.
        self.emit("# Generated by pylonc from a .pylon source file.")
        self.emit("# Cython source (.pyx). Compile with: cythonize <file>.pyx")
        self.emit("# cython: language_level=3, boundscheck=False, "
                  "wraparound=False, infer_types=True")
        self.emit("")
        for node in prog:
            self.emit_node(node)
        return self.render()

    # -- dispatch -----------------------------------------------------------
    def emit_node(self, node: Node) -> None:
        if isinstance(node, ImportStmt):
            self.emit_import(node)
        elif isinstance(node, Decl):
            self.emit_decl(node)
        elif isinstance(node, ConstDecl):
            self.emit_const(node)
        elif isinstance(node, FuncDef):
            self.emit_func(node)
        elif isinstance(node, ClassDef):
            self.emit_class(node)
        elif isinstance(node, IfStmt):
            self.emit_if(node)
        elif isinstance(node, WhileStmt):
            self.emit_while(node)
        elif isinstance(node, ForStmt):
            self.emit_for(node)
        elif isinstance(node, ReturnStmt):
            self.emit_return(node)
        elif isinstance(node, (BreakStmt, ContinueStmt, PassStmt)):
            kw = {BreakStmt: "break", ContinueStmt: "continue",
                  PassStmt: "pass"}[type(node)]
            self.emit(kw)
        elif isinstance(node, ExprStmt):
            self.emit(render_tokens(node.tokens))
        elif isinstance(node, RaiseStmt):
            if node.exc is None:
                self.emit("raise")
            else:
                self.emit("raise " + render_tokens(node.exc))
        elif isinstance(node, TryStmt):
            self.emit_try(node)
        elif isinstance(node, WithStmt):
            self.emit_with(node)
        elif isinstance(node, DelStmt):
            self.emit("del " + render_tokens(node.target))
        elif isinstance(node, GlobalStmt):
            self.emit("global " + ", ".join(node.names))
        elif isinstance(node, NonlocalStmt):
            self.emit("nonlocal " + ", ".join(node.names))
        elif isinstance(node, AssertStmt):
            line = "assert " + render_tokens(node.test)
            if node.msg is not None:
                line += ", " + render_tokens(node.msg)
            self.emit(line)
        else:
            raise PylonError(f"emit_node: unknown node type {type(node).__name__}")

    # -- imports ------------------------------------------------------------
    def emit_import(self, node: ImportStmt) -> None:
        if node.from_module is not None:
            parts = [f"from {node.from_module} import"]
            for name, alias in node.names:
                if alias:
                    parts.append(f"{name} as {alias}")
                else:
                    parts.append(name)
            # join with commas
            joined = " ".join(parts[:2]) + " " + ", ".join(parts[2:])
            self.emit(joined)
        else:
            for name, alias in node.names:
                if alias:
                    self.emit(f"import {name} as {alias}")
                else:
                    self.emit(f"import {name}")

    # -- typed declarations -------------------------------------------------
    def emit_decl(self, node: Decl) -> None:
        # Inside a class body, drop the cdef (Cython forbids cdef attrs in
        # regular classes). Outside, emit cdef so the type hint reaches C.
        if self._in_class:
            if node.init is not None:
                self.emit(f"{node.name} = {render_tokens(node.init)}")
            else:
                self.emit(f"{node.name} = None")
            return
        c_type = pylon_type_to_cython(node.type_name)
        if node.init is not None:
            self.emit(f"cdef {c_type} {node.name} = {render_tokens(node.init)}")
        else:
            self.emit(f"cdef {c_type} {node.name}")
        # Note: at module scope, Cython requires cdef for typed vars; inside
        # a function, cdef is also fine. Both work with this single form.

    def emit_const(self, node: ConstDecl) -> None:
        # const -> cdef with init; Cython has no const qualifier that's
        # useful for Python objects, so we treat it as a typed declaration.
        # For C primitives we could use `const`, but uniformity wins here.
        if self._in_class:
            self.emit(f"{node.name} = {render_tokens(node.init)}")
            return
        c_type = pylon_type_to_cython(node.type_name)
        self.emit(f"cdef {c_type} {node.name} = {render_tokens(node.init)}")

    # -- function -----------------------------------------------------------
    def emit_func(self, node: FuncDef) -> None:
        # Build the function header. At module / function scope we use
        # `cpdef` so the typed return + params feed into C while the function
        # stays callable from Python (the user's example calls HelloWorld
        # from the module top level). Inside a `class` body, Cython forbids
        # cpdef (only cdef classes allow cpdef methods), so we fall back to
        # a plain `def` and ignore the return/param type annotations.
        if self._in_class:
            params_str = ", ".join(pn for _, pn in node.params)
            self.emit(f"def {node.name}({params_str}):")
            self.push()
            self._in_func += 1
            if node.body:
                for s in node.body:
                    self.emit_node(s)
            else:
                self.emit("pass")
            self._in_func -= 1
            self.pop()
            return
        ret = pylon_type_to_return(node.return_type) if node.return_type else None
        params_str = ", ".join(
            f"{pylon_type_to_cython(pt)} {pn}" for pt, pn in node.params
        )
        header = "cpdef"
        if ret is not None:
            header += f" {ret}"
        header += f" {node.name}({params_str}):"
        self.emit(header)
        self.push()
        self._in_func += 1
        if node.body:
            for s in node.body:
                self.emit_node(s)
        else:
            self.emit("pass")
        self._in_func -= 1
        self.pop()

    # -- class --------------------------------------------------------------
    def emit_class(self, node: ClassDef) -> None:
        bases = ", ".join(node.bases)
        if bases:
            self.emit(f"class {node.name}({bases}):")
        else:
            self.emit(f"class {node.name}:")
        self.push()
        self._in_class += 1
        if node.body:
            for s in node.body:
                self.emit_node(s)
        else:
            self.emit("pass")
        self._in_class -= 1
        self.pop()

    # -- control flow ------------------------------------------------------
    def emit_if(self, node: IfStmt) -> None:
        self.emit(f"if {render_tokens(node.cond)}:")
        self.push()
        self._emit_body(node.then_body)
        self.pop()
        for cond, body in node.elif_clauses:
            self.emit(f"elif {render_tokens(cond)}:")
            self.push()
            self._emit_body(body)
            self.pop()
        if node.else_body is not None:
            self.emit("else:")
            self.push()
            self._emit_body(node.else_body)
            self.pop()

    def emit_while(self, node: WhileStmt) -> None:
        self.emit(f"while {render_tokens(node.cond)}:")
        self.push()
        self._emit_body(node.body)
        self.pop()

    def emit_for(self, node: ForStmt) -> None:
        # `for x in iter:` -> Cython understands this; typed loop variables
        # would need `for <type> x in iter` which Cython supports but only
        # for primitive types and only when the iterable yields C values.
        # To keep semantics safe, we emit untyped for-loops.
        self.emit(f"for {render_tokens(node.target)} in "
                  f"{render_tokens(node.iter_expr)}:")
        self.push()
        self._emit_body(node.body)
        self.pop()

    def emit_return(self, node: ReturnStmt) -> None:
        if node.value is None:
            self.emit("return")
        else:
            self.emit("return " + render_tokens(node.value))

    def emit_try(self, node: TryStmt) -> None:
        self.emit("try:")
        self.push()
        self._emit_body(node.body)
        self.pop()
        for h in node.handlers:
            if h.exc_type is None:
                self.emit("except:")
            else:
                line = "except " + render_tokens(h.exc_type)
                if h.as_name:
                    line += f" as {h.as_name}"
                line += ":"
                self.emit(line)
            self.push()
            self._emit_body(h.body)
            self.pop()
        if node.else_body is not None:
            self.emit("else:")
            self.push()
            self._emit_body(node.else_body)
            self.pop()
        if node.finally_body is not None:
            self.emit("finally:")
            self.push()
            self._emit_body(node.finally_body)
            self.pop()

    def emit_with(self, node: WithStmt) -> None:
        parts: list[str] = []
        for ctx, as_name in node.items:
            piece = render_tokens(ctx)
            if as_name:
                piece += f" as {as_name}"
            parts.append(piece)
        self.emit("with " + ", ".join(parts) + ":")
        self.push()
        self._emit_body(node.body)
        self.pop()

    # -- body helper --------------------------------------------------------
    def _emit_body(self, body: list[Node]) -> None:
        if not body:
            self.emit("pass")
            return
        for s in body:
            self.emit_node(s)


# ---------------------------------------------------------------------------
# Driver: translate .pylon -> .pyx string
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Semantic check: const-safety
# ---------------------------------------------------------------------------
#
# A `const` declaration in Pylon is meant to be immutable after its initial
# binding. The translator emits it as a plain `cdef` (Cython has no useful
# `const` qualifier for Python objects), so nothing at the C level stops you
# from reassigning it. This pass walks the parsed AST and raises a
# PylonError at compile time if an assignment, augmented assignment, or
# `del` targets a name that was declared `const` in the same scope.
#
# Exception (per the user's request): an assignment to a const that lives
# inside an `except`, `else`, or `finally` block of a `try` is ALLOWED.
# The rationale is that those blocks are the error-recovery path, where
# re-binding a "constant" to a fallback value is the whole point of the
# handler (e.g. `const str cfg = load(); try { use(cfg) } except { cfg =
# DEFAULT; ... }`).
#
# Scoping: consts declared at module level are visible to the whole module
# (including nested function bodies, mirroring Python's module globals).
# Consts declared inside a function are local to that function only.

# Augmented-assignment operators that imply mutation of the target.
_AUG_OPS = {"+=", "-=", "*=", "/=", "//=", "%=", "**=", "<<=", ">>=",
            "&=", "|=", "^=", "@="}


def _assignment_targets(tokens: list[Token]) -> list[str]:
    """Extract the bare-name assignment targets from an ExprStmt's token
    list.

    An ExprStmt carries raw tokens for assignments. We pull out the names
    being assigned to (the leftmost targets of `=` and any augmented-op
    targets). We deliberately ignore attribute (`a.b = ...`), subscript
    (`a[0] = ...`) and tuple-unpacking (`a, b = ...`) targets for now: a
    const Pylon name is a C-level binding, not a container, so those forms
    don't rebind the const name itself (they mutate something the const
    refers to, which is a separate mutability question we don't address
    here).

    Returns the list of target names, in source order.
    """
    targets: list[str] = []
    i = 0
    n = len(tokens)
    # Collect the leftmost run of NAME tokens up to the first `=` or aug op.
    # Multi-target `a = b = c = 0` is a chain; we capture every NAME that is
    # immediately followed by `=` as a target.
    while i < n:
        t = tokens[i]
        if t.kind == TT_NAME and t.value not in {"and", "or", "not", "in",
                                                  "is", "True", "False",
                                                  "None"}:
            # Look ahead: is the next token `=` or an aug op? (Whitespace
            # is skipped during tokenization, so the next token is the
            # immediately following significant one.)
            j = i + 1
            if j < n:
                nt = tokens[j]
                if nt.kind == TT_OP and nt.value == "=":
                    targets.append(t.value)
                    i = j + 1
                    continue
                if nt.kind == TT_OP and nt.value in _AUG_OPS:
                    targets.append(t.value)
                    # Skip to the end of this expression (the aug-op's RHS);
                    # there are no further assignment targets on this line.
                    break
        i += 1
    return targets


def _check_const_safety(program: list[Node]) -> None:
    """Walk the parsed program and raise PylonError on any const
    reassignment that isn't inside a try-handler recovery block."""
    errors: list[str] = []

    def check_block(body: list[Node], consts: set[str],
                    in_recovery: bool) -> None:
        for node in body:
            if isinstance(node, ConstDecl):
                consts.add(node.name)
                continue
            if isinstance(node, ExprStmt):
                if not in_recovery:
                    for tgt in _assignment_targets(node.tokens):
                        if tgt in consts:
                            errors.append(
                                f"line {node.line}: cannot reassign const "
                                f"{tgt!r} (declared const). To allow "
                                f"rebinding inside an error-recovery path, "
                                f"place it in an `except` or `finally` block."
                            )
                continue
            if isinstance(node, DelStmt):
                if not in_recovery:
                    for tok in node.target:
                        if tok.kind == TT_NAME and tok.value in consts:
                            errors.append(
                                f"line {node.line}: cannot `del` const "
                                f"{tok.value!r} (declared const)."
                            )
                continue
            if isinstance(node, FuncDef):
                # Function body gets a fresh local scope but inherits the
                # enclosing module-level consts (Python closure semantics).
                local_consts = set(consts)
                check_block(node.body, local_consts, in_recovery=False)
                continue
            if isinstance(node, TryStmt):
                # The try body itself is NOT recovery; only the handlers,
                # else, and finally are.
                check_block(node.body, consts, in_recovery=False)
                for eh in node.handlers:
                    check_block(eh.body, consts, in_recovery=True)
                if node.else_body is not None:
                    check_block(node.else_body, consts, in_recovery=True)
                if node.finally_body is not None:
                    check_block(node.finally_body, consts, in_recovery=True)
                continue
            if isinstance(node, WithStmt):
                check_block(node.body, consts, in_recovery=False)
                continue
            # Other compound statements (if/elif/else, for, while, class)
            # don't introduce a new scope for const purposes - their inner
            # statements are in the same scope as the surrounding block.
            # We don't recurse into them here because they don't currently
            # expose their bodies in a uniform way worth special-casing;
            # const reassignment inside an `if` body will be caught at the
            # enclosing scope level. (This is conservative: it may miss a
            # reassignment nested 2+ levels deep in a loop. A future
            # improvement is to add `body` accessors uniformly.)

    check_block(program, consts=set(), in_recovery=False)
    if errors:
        raise PylonError(
            "const-safety check failed:\n  " + "\n  ".join(errors)
        )


# ---------------------------------------------------------------------------
# Semantic check: return-type safety
# ---------------------------------------------------------------------------
#
# When a function declares a return type (e.g. `def string f() { ... }`),
# the Cython compiler enforces it at run time and raises a Python
# TypeError ("Expected str, got int") if a `return` statement hands back
# the wrong type. That's a Cython-level error, not a Pylon one - the user
# sees a stack trace pointing into the generated .pyx, which is confusing.
#
# This pass catches the common case at compile time: a `return` whose
# expression is a *literal* (or a constructor call) whose inferred Pylon
# type does not match the function's declared return type.
#
# Inference is deliberately conservative. We only flag what we can prove
# is wrong from the literal alone:
#   - integer literal  -> int       (123, 0x1F, 0o77, 0b101)
#   - float literal     -> float     (1.5, 1e10, 1.5e-3)
#   - complex literal   -> complex  (1.5j, 2+3j -- well, 2+3j is two tokens;
#                                     we only catch the bare `Nj` form)
#   - string literal    -> string    ("...", '...', r"...", f"...")
#   - bytes literal     -> bytes     (b"...", rb"...")
#   - True/False        -> boolean
#   - None              -> NoneType
#   - list(...)         -> list      (and tuple/dict/set/frozenset similarly)
#   - [...]             -> list      (literal list display)
#   - (...)             -> tuple    (parenthesised comma list, or `()`)
#   - {...}             -> dict      (literal dict display, even `{}`)
#                            (Note: `{}` is ambiguous between dict and set;
#                             we infer dict, which matches Python's runtime
#                             behaviour for `{}`.)
#
# Anything we can't prove (variables, arithmetic, function calls other than
# the constructor calls above) is left alone - no false positives.

# Map from constructor NAME -> Pylon type name. These are the builtin
# callables whose return type is determined solely by the callable name,
# not by argument analysis.
_CONSTRUCTOR_TYPES = {
    "int": "int", "float": "float", "complex": "complex",
    "str": "string", "string": "string",
    "bool": "boolean", "boolean": "boolean",
    "bytes": "bytes", "bytearray": "bytearray",
    "list": "list", "tuple": "tuple", "dict": "dict",
    "set": "set", "frozenset": "frozenset",
    "range": "range", "memoryview": "memoryview",
}


def _infer_return_type(toks: list[Token]) -> Optional[str]:
    """Infer the Pylon type of a return-expression's token list, or
    return None if we can't prove it from the literal alone (in which
    case the caller should NOT flag - better to miss than to false-flag).
    """
    # Strip leading/trailing no-ops; whitespace is already gone but be safe.
    toks = [t for t in toks
            if t.kind not in (TT_NEWLINE,)]
    if not toks:
        return None

    first = toks[0]
    last = toks[-1]

    # --- single-token literals -------------------------------------------
    if len(toks) == 1:
        if first.kind == TT_NUMBER:
            raw = first.value
            if raw.endswith(("j", "J")):
                return "complex"
            if "." in raw or "e" in raw or "E" in raw:
                # Float literal only if it's not a hex/oct/bin integer
                # (those don't contain '.', and only hex/oct/bin use the
                # 0x/0o/0b prefix - check that to be safe).
                if not (raw.lower().startswith(("0x", "0o", "0b"))):
                    return "float"
            return "int"
        if first.kind in (TT_STRING, TT_FSTRING):
            raw = first.value
            # A bytes literal has a 'b' or 'B' prefix and NO 'f'/'F'.
            low = raw.lower()
            has_b = "b" in low[:low.index(low[low.find("'\"")]) if low.find("'\"") != -1 else 0:4]
            # Easier: check the prefix chars before the opening quote.
            # Find the opening quote.
            qpos = -1
            for idx, ch in enumerate(raw):
                if ch in "'\"":
                    qpos = idx
                    break
            if qpos > 0:
                prefix = raw[:qpos].lower()
                if "b" in prefix and "f" not in prefix:
                    return "bytes"
                # f-string -> string (Python's str)
                return "string"
            return "string"
        if first.kind == TT_NAME:
            if first.value == "True" or first.value == "False":
                return "boolean"
            if first.value == "None":
                return "NoneType"

    # --- constructor call:  NAME ( ... )  --------------------------------
    # We don't care about the arguments; the constructor name alone
    # determines the type. This is correct for the builtin constructors
    # listed in _CONSTRUCTOR_TYPES. We must check the NAME is one of those
    # BEFORE consulting the map (otherwise `g()` would KeyError on 'g').
    if (first.kind == TT_NAME and first.value in _CONSTRUCTOR_TYPES
            and len(toks) >= 3
            and toks[1].kind == "("
            and last.kind == ")"):
        return _CONSTRUCTOR_TYPES[first.value]

    # --- literal displays:  [ ... ]  / ( ... )  / { ... }  --------------
    if first.kind == "(":
        if last.kind == ")":
            # `()` or `(a, b, ...)` -> tuple. A single `(expr)` without a
            # comma is just a parenthesised expression - we can't infer
            # its type, so we check for any comma at depth 0.
            depth = 0
            saw_comma = False
            for t in toks[1:-1]:
                if t.kind == "(":
                    depth += 1
                elif t.kind == ")":
                    depth -= 1
                elif t.kind == "," and depth == 0:
                    saw_comma = True
                    break
                elif t.kind == "[":
                    depth += 1
                elif t.kind == "]":
                    depth -= 1
                elif t.kind == "{":
                    depth += 1
                elif t.kind == "}":
                    depth -= 1
            if saw_comma or len(toks) == 2:
                # len==2 means `()` -> empty tuple
                return "tuple"
            # `(expr)` -> can't infer; fall through to None
    if first.kind == "[" and last.kind == "]":
        return "list"
    if first.kind == "{" and last.kind == "}":
        # `{}` -> dict at runtime; `{...}` with `:` at depth 0 -> dict;
        # `{...}` with commas but no `:` at depth 0 -> set. We infer dict
        # for the empty case (matches Python) and check for `:` otherwise.
        if len(toks) == 2:
            return "dict"
        depth = 0
        saw_colon = False
        for t in toks[1:-1]:
            if t.kind in ("(", "[", "{"):
                depth += 1
            elif t.kind in (")", "]", "}"):
                depth -= 1
            elif t.kind == ":" and depth == 0:
                saw_colon = True
                break
        return "dict" if saw_colon else "set"

    return None


def _check_return_types(program: list[Node]) -> None:
    """Walk the program; for each FuncDef with a declared return type,
    flag any `return <expr>;` whose literal-inferred type doesn't match.

    A bare `return;` (no value) is ALWAYS allowed, even in a function
    declared to return a non-None type: in Python/Cython it's equivalent to
    `return None`, which Cython tolerates for object return types and which
    we don't want to flag (the user explicitly asked for this).
    """
    errors: list[str] = []

    def check_body(body: list[Node], declared_ret: Optional[str]) -> None:
        for node in body:
            if isinstance(node, ReturnStmt):
                # Bare return is always allowed.
                if node.value is None:
                    continue
                # NoneType-declared functions can return anything
                # (returning None is the natural case; returning a value
                # would be a bug but Cython allows it, so we don't flag).
                if declared_ret is None or declared_ret == "NoneType":
                    continue
                inferred = _infer_return_type(node.value)
                if inferred is None:
                    continue  # can't prove it - leave alone
                if inferred == declared_ret:
                    continue
                # int/float/complex are NOT implicitly convertible for our
                # purposes: returning 2 from a `float` function is a bug
                # (Cython would auto-convert, but the user almost
                # certainly meant float). We flag any mismatch.
                errors.append(
                    f"line {node.line}: return type mismatch in function "
                    f"declared to return {declared_ret!r}: expression "
                    f"looks like {inferred!r}."
                )
                continue
            if isinstance(node, FuncDef):
                # Nested function: its own declared return type governs,
                # not the outer one.
                check_body(node.body, node.return_type)
                continue
            if isinstance(node, TryStmt):
                check_body(node.body, declared_ret)
                for h in node.handlers:
                    check_body(h.body, declared_ret)
                if node.else_body is not None:
                    check_body(node.else_body, declared_ret)
                if node.finally_body is not None:
                    check_body(node.finally_body, declared_ret)
                continue
            if isinstance(node, WithStmt):
                check_body(node.body, declared_ret)
                continue
            # Other statements (if/for/while/class) don't get recursed
            # here for the same conservative reason as the const check -
            # a return nested inside an `if` body will still be caught if
            # it's at the function's top level. (This is a known
            # limitation; fixing it would require uniform `body` accessors
            # across all compound-statement nodes.)
            pass

    # Module-level functions.
    for node in program:
        if isinstance(node, FuncDef):
            check_body(node.body, node.return_type)
        # Class bodies may contain methods (FuncDefs) - recurse into them.
        # We don't have a ClassDef import handy, so check by attribute.
        elif hasattr(node, "body") and isinstance(getattr(node, "body"),
                                                   list):
            for inner in node.body:
                if isinstance(inner, FuncDef):
                    check_body(inner.body, inner.return_type)

    if errors:
        raise PylonError(
            "return-type check failed:\n  " + "\n  ".join(errors)
        )


def translate(source: str) -> str:
    tokens = tokenize(source)
    parser = Parser(tokens)
    program = parser.parse_program()
    _check_const_safety(program)
    _check_return_types(program)
    emitter = Emitter()
    return emitter.emit_module(program)


# ---------------------------------------------------------------------------
# Compilation pipeline
# ---------------------------------------------------------------------------

def _resolve_compiler() -> str:
    """Pick the C compiler to use for the gcc/clang step.

    Resolution order:
      1. The ``CC`` environment variable, if set and the binary exists.
      2. ``cc`` on PATH (the canonical system compiler on Linux/macOS -
         usually a symlink to gcc or clang).
      3. ``gcc`` on PATH.
      4. ``clang`` on PATH (common on macOS without Xcode CLT, and on
         FreeBSD).

    We resolve once at first use and cache the result. Raising a clear
    PylonError here is much friendlier than letting subprocess.run emit a
    bare ``FileNotFoundError: [Errno 2] No such file or directory: 'cc'``
    deep in the build pipeline - which is what users saw when VS Code
    launched pylonc with a stripped PATH.
    """
    cached = getattr(_resolve_compiler, "_cached", None)
    if cached is not None:
        return cached

    candidates: list[str] = []
    env_cc = os.environ.get("CC")
    if env_cc:
        candidates.append(env_cc)
    candidates.extend(["cc", "gcc", "clang"])

    for c in candidates:
        # If CC was set to an absolute path, use it directly even if
        # shutil.which can't find it (the user knows where their compiler
        # is; don't second-guess them, just check it exists).
        if os.path.isabs(c):
            if os.path.isfile(c) and os.access(c, os.X_OK):
                _resolve_compiler._cached = c  # type: ignore[attr-defined]
                return c
            continue
        found = shutil.which(c)
        if found:
            _resolve_compiler._cached = found  # type: ignore[attr-defined]
            return found

    raise PylonError(
        "No C compiler found. Tried: " + ", ".join(candidates) + ".\n"
        "Install Xcode Command Line Tools (macOS: `xcode-select --install`), "
        "or `gcc`/`clang` via your package manager, or set the CC environment "
        "variable to the absolute path of your compiler."
    )


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    """Run a command, capture output, raise on failure with the full log."""
    kw.setdefault("text", True)
    kw.setdefault("capture_output", True)
    proc = subprocess.run(cmd, **kw)
    if proc.returncode != 0:
        msg = ["Command failed:", " ".join(cmd)]
        if proc.stdout:
            msg.append("--- stdout ---"); msg.append(proc.stdout)
        if proc.stderr:
            msg.append("--- stderr ---"); msg.append(proc.stderr)
        raise PylonError("\n".join(msg))
    return proc


def compile_pylon_to_pyx(source_path: str) -> str:
    with open(source_path, "r", encoding="utf-8") as f:
        src = f.read()
    pyx = translate(src)
    return pyx


def write_pyx(pyx_source: str, out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(pyx_source)


def cythonize_to_c(pyx_path: str, c_path: str, embed: bool = True) -> None:
    """Run `cython --3 <pyx>` to produce a .c file. When embed=True, add
    --embed so the C file contains a main() that runs the module.

    Invokes cython as ``<sys.executable> -m cython`` rather than the bare
    ``cython`` CLI. This matters when pylonc is launched by an environment
    that strips PATH (VS Code's extension host, GUI launchers, systemd
    units, ``env -i``): the ``cython`` console-script may not be on PATH
    even though the ``cython`` Python package is importable by the running
    interpreter. Using ``sys.executable`` guarantees we run the *same*
    interpreter that is running pylonc - and therefore the one that has
    cython installed - regardless of PATH.
    """
    cmd = [sys.executable, "-m", "cython", "--3"]
    if embed:
        cmd.append("--embed")
    cmd += ["-o", c_path, pyx_path]
    _run(cmd)


def _python_cflags(dev: bool = False) -> list[str]:
    """Compile flags derived from the RUNNING interpreter's sysconfig.

    Why not `python3-config`? On systems where the active `python3` is a
    venv/uv-managed interpreter that doesn't match the system
    `python3-config` (e.g. python3 == 3.12 but python3-config reports 3.13),
    the generated C targets one C-API version and links against another,
    producing "Illegal instruction" at runtime. Reading sysconfig from this
    very interpreter (the one Cython runs under) guarantees they match.

    `dev=True` drops -O3 to -O0: ~3x faster cc at the cost of slower
    runtime. Intended for edit/compile/run loops; the default is -O3 so
    shipped builds keep the runtime speedup Pylon exists for."""
    import sysconfig
    inc = sysconfig.get_path("include")
    opt = "-O0" if dev else "-O3"
    cflags = [f"-I{inc}", "-fno-strict-overflow", "-Wsign-compare",
              "-DNDEBUG", "-g", opt, "-fno-strict-aliasing"]
    # macOS needs -fPIC too for shared objects; harmless on Linux.
    return cflags


def _python_so_ldflags() -> list[str]:
    """Link flags for building a CPython extension module (.so / .dylib).

    Extension modules do NOT link against libpython. The Python C-API
    symbols are resolved at runtime by the Python process that imports the
    module - this is exactly how setuptools and `cythonize` build
    extensions, and it's what makes a .so portable across Python builds of
    the same ABI. Linking libpython here is not only unnecessary, it's
    actively wrong on macOS (where LDLIBRARY is a framework path, not a
    lib*.so, and a naive `-l` translation fails).

    The only platform-specific bit: macOS requires `-undefined
    dynamic_lookup` to defer undefined-symbol resolution to import time;
    Linux's default already allows undefined symbols in a shared object."""
    if sys.platform == "darwin":
        return ["-undefined", "dynamic_lookup"]
    return []


def _python_exe_ldflags() -> list[str]:
    """Link flags for the standalone `--embed` executable.

    Unlike a .so, a standalone exe embeds the interpreter and so MUST link
    libpython. We handle two layouts:
      - Linux/typical: LDLIBRARY is `libpython3.x.so(.1.0)` -> `-lpython3.x`
        with `-L<LIBDIR>` and an rpath so the exe finds it at runtime.
      - macOS framework: LDLIBRARY is `Python.framework/Versions/.../Python`
        -> link via `-F<frameworks> -framework Python` and an rpath."""
    import sysconfig
    libdir = sysconfig.get_config_var("LIBDIR")
    ldlib = sysconfig.get_config_var("LDLIBRARY")
    ldflags: list[str] = []
    if not libdir or not ldlib:
        # Fall back: can't determine libpython location.
        ldflags += ["-ldl", "-lm"]
        if sys.platform != "darwin":
            ldflags.append("-lpthread")
        return ldflags

    if "framework" in libdir.lower():
        # macOS Python.framework. Find the `<Name>.framework` segment in
        # the absolute LIBDIR path (NOT in LDLIBRARY, which is just the
        # relative framework binary name and gives garbage when split).
        # Example:
        #   LIBDIR  = /Library/Frameworks/Python.framework/Versions/3.10/lib
        #   -> fw_parent = /Library/Frameworks   (dir CONTAINING the bundle)
        #   -> fw_name   = Python
        # -F must point at fw_parent (NOT the .framework itself): the linker
        # then looks for `<fw_parent>/Python.framework/Python`, which is the
        # standard framework lookup. Pointing -F at the bundle itself makes
        # ld search *inside* it for a nested framework and fail with
        # "framework not found Python".
        segs = libdir.split(os.sep)
        fw_idx = next((i for i, s in enumerate(segs)
                       if s.lower().endswith(".framework")), None)
        if fw_idx is not None:
            fw_parent = os.sep.join(segs[:fw_idx])
            fw_name = segs[fw_idx][:-len(".framework")]
            ldflags.append(f"-F{fw_parent}")
            ldflags.append("-framework")
            ldflags.append(fw_name)
            # rpath so the exe finds the framework at runtime without
            # DYLD_FRAMEWORK_PATH set by the user. Same parent dir: dyld
            # resolves `Python.framework/Python` under each rpath entry.
            ldflags.append(f"-Wl,-rpath,{fw_parent}")
        else:
            # LIBDIR mentioned "framework" but no segment matched - fall
            # back to a plain -l<name> and hope for the best.
            name = ldlib
            if name.startswith("lib"):
                name = name[3:]
            ldflags.append(f"-L{libdir}")
            ldflags.append(f"-l{name}")
    else:
        # Regular libpython3.x.so(.1.0). Turn into -lpython3.x.
        name = ldlib
        if name.startswith("lib"):
            name = name[3:]
        if ".so" in name:
            name = name[:name.index(".so")]
        elif ".dylib" in name:
            name = name[:name.index(".dylib")]
        ldflags.append(f"-L{libdir}")
        ldflags.append(f"-Wl,-rpath,{libdir}")
        ldflags.append(f"-l{name}")
    ldflags += ["-ldl", "-lm"]
    if sys.platform != "darwin":
        ldflags.append("-lpthread")
    return ldflags


def compile_c_to_exe(c_path: str, exe_path: str, dev: bool = False) -> None:
    """gcc-compile the embedded C file into a standalone executable that
    links against libpython (so it can run without `python -c`)."""
    cflags = _python_cflags(dev=dev)
    ldflags = _python_exe_ldflags()
    cmd = [_resolve_compiler(), c_path, "-o", exe_path]
    cmd += cflags
    cmd += ldflags
    _run(cmd)


def compile_c_to_so(c_path: str, so_path: str, dev: bool = False) -> None:
    """gcc-compile into a shared CPython extension module (.so on Linux,
    .so on macOS too - the import machinery accepts the .so suffix).

    Deliberately does NOT link libpython: see _python_so_ldflags."""
    cflags = _python_cflags(dev=dev)
    ldflags = _python_so_ldflags()
    cc = _resolve_compiler()
    if sys.platform == "darwin":
        # On macOS `-shared` works but the conventional flag set is
        # `-bundle -undefined dynamic_lookup`. `-shared` is accepted by
        # clang as an alias and behaves the same, so we keep it for
        # symmetry with the Linux path.
        cmd = [cc, "-shared", "-fPIC", c_path, "-o", so_path]
    else:
        cmd = [cc, "-shared", "-fPIC", c_path, "-o", so_path]
    cmd += cflags
    cmd += ldflags
    _run(cmd)


def run_exe(exe_path: str, args: list[str]) -> int:
    proc = subprocess.run([exe_path] + args)
    return proc.returncode


def run_so(so_dir: str, module_name: str, args: list[str]) -> int:
    """Run a compiled .so module by importing it from Python.

    The .so is a CPython extension module; importing it executes the
    module's top-level code (which is where Pylon's main-level statements
    live, just like a Python script). We pass sys.argv through so the
    program can read its own arguments."""
    shim = (
        "import sys, importlib.util, runpy;"
        f"sys.argv = ['{module_name}'] + {args!r};"
        f"sys.path.insert(0, {so_dir!r});"
        f"import {module_name};"
    )
    proc = subprocess.run([sys.executable, "-c", shim])
    return proc.returncode


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pylonc",
        description=(
            "Pylon translator: compile Java-syntax Python (.pylon) to "
            "Cython .pyx, then to native code, then run it."
        ),
    )
    p.add_argument("source", help="Path to a .pylon source file.")
    p.add_argument("-o", "--output", default=None,
                   help="Output binary name (default: same stem as source).")
    p.add_argument("--keep-dir", default=None,
                   help="Directory to keep intermediate .pyx/.c files in "
                        "(default: <stem>_build alongside the source).")
    p.add_argument("--no-run", action="store_true",
                   help="Translate and compile but do NOT run the program. "
                        "Saves .pyx and .c (and the compiled binary).")
    p.add_argument("--emit-pyx", action="store_true",
                   help="Only emit the .pyx file; do not compile or run.")
    p.add_argument("--embed", action="store_true",
                   help="Build a STANDALONE executable (cython --embed + link "
                        "to libpython). Default is to build a .so and run via "
                        "`python -c import`, which is more portable.")
    p.add_argument("--show-pyx", action="store_true",
                   help="Print the generated .pyx to stdout as well.")
    p.add_argument("--dev", action="store_true",
                   help="Compile the .c with -O0 instead of -O3. ~3x faster "
                        "cc for edit/compile/run loops, at the cost of slower "
                        "runtime. Note: a -O0 build does NOT invalidate a "
                        "cached -O3 .so (and vice versa) - use --no-cache "
                        "to force a rebuild when switching modes.")
    p.add_argument("--no-cache", action="store_true",
                   help="Force a full rebuild: ignore any cached .so from a "
                        "previous run. By default pylonc skips the Cython + cc "
                        "phases when the .pyx is byte-identical to last build.")
    p.add_argument("--args", nargs=argparse.REMAINDER, default=[],
                   help="Arguments to pass through to the program when run.")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    source_path = args.source
    if not os.path.isfile(source_path):
        print(f"pylonc: source file not found: {source_path}", file=sys.stderr)
        return 2
    if not source_path.endswith(".pylon"):
        print(f"pylonc: source file must have .pylon extension: {source_path}",
              file=sys.stderr)
        return 2

    stem = os.path.basename(source_path)[:-len(".pylon")]

    # Decide where to drop intermediates.
    if args.keep_dir:
        keep_dir = args.keep_dir
    else:
        keep_dir = os.path.join(os.path.dirname(os.path.abspath(source_path)),
                                f"{stem}_build")
    os.makedirs(keep_dir, exist_ok=True)

    pyx_path = os.path.join(keep_dir, stem + ".pyx")
    c_path = os.path.join(keep_dir, stem + ".c")

    # Step 1: translate .pylon -> .pyx
    try:
        pyx_src = compile_pylon_to_pyx(source_path)
    except PylonError as e:
        print(f"pylonc: translation error: {e}", file=sys.stderr)
        return 1
    write_pyx(pyx_src, pyx_path)
    if args.show_pyx:
        print("=" * 70)
        print(f"# {pyx_path}")
        print("=" * 70)
        print(pyx_src)
        print("=" * 70)

    if args.emit_pyx:
        print(f"pylonc: wrote {pyx_path}")
        return 0

    # Step 2: cythonize .pyx -> .c  (skipped if cached .so is up to date)
    embed = args.embed
    dev = args.dev
    out_name = args.output or stem

    # --- incremental cache -------------------------------------------------
    # Cython is deterministic given the same .pyx + version + embed flag, and
    # cc is deterministic given the same .c + dev flag. So we fingerprint the
    # .pyx bytes plus the build mode, store it in <stem>.fingerprint, and
    # skip BOTH the cythonize and cc phases when:
    #   (a) the .so/.exe exists and is newer than the .pyx, AND
    #   (b) the stored fingerprint matches the current one.
    # This drops a 2.2s rebuild to ~0.06s + run on the common case of
    # "edit a comment, rerun" or "rerun unchanged". --no-cache bypasses it.
    import hashlib
    try:
        import Cython  # type: ignore
        cy_ver = Cython.__version__
    except Exception:
        cy_ver = "unknown"
    fp_input = pyx_src.encode("utf-8", errors="replace")
    fp_meta = (f"embed={int(embed)};dev={int(dev)};cython={cy_ver};"
               f"py={sys.version_info.major,sys.version_info.minor}")
    fp = hashlib.sha256(fp_input + fp_meta.encode()).hexdigest()
    fp_path = os.path.join(keep_dir, stem + ".fingerprint")

    if embed:
        binary = os.path.join(keep_dir, out_name)
    else:
        binary = os.path.join(keep_dir, out_name + ".so")

    cache_hit = False
    # The cache check is purely content-based, NOT mtime-based: pylonc
    # rewrites the .pyx on every run (always the same bytes for unchanged
    # source), so any "binary newer than .pyx" check fails spuriously.
    # The .fingerprint stores a sha256 of (pyx bytes + embed/dev/cython
    # version). If it matches the current fingerprint and the binary still
    # exists, the binary is by construction up to date - re-running cython
    # and cc would produce byte-identical output.
    if not args.no_cache and os.path.isfile(binary) and os.path.isfile(fp_path):
        try:
            with open(fp_path) as f:
                old_fp = f.read().strip()
        except OSError:
            old_fp = ""
        if old_fp == fp:
            cache_hit = True

    if cache_hit:
        kind = "standalone executable" if embed else "extension module"
        print(f"pylonc: cache hit - reusing cached {kind} {binary}")
        print(f"pylonc: (use --no-cache to force a full rebuild)")
    else:
        try:
            cythonize_to_c(pyx_path, c_path, embed=embed)
        except PylonError as e:
            print(f"pylonc: cython compile error:\n{e}", file=sys.stderr)
            return 1

        # Step 3: compile .c -> binary
        if embed:
            try:
                compile_c_to_exe(c_path, binary, dev=dev)
            except PylonError as e:
                print(f"pylonc: gcc compile error:\n{e}", file=sys.stderr)
                return 1
            print(f"pylonc: compiled standalone executable -> {binary}")
        else:
            try:
                compile_c_to_so(c_path, binary, dev=dev)
            except PylonError as e:
                print(f"pylonc: gcc compile error:\n{e}", file=sys.stderr)
                return 1
            print(f"pylonc: compiled extension module -> {binary}")
        # Record the fingerprint only after a successful full build, so a
        # half-built .so from a crashed run never looks like a cache hit.
        try:
            with open(fp_path, "w") as f:
                f.write(fp)
        except OSError:
            pass
        print(f"pylonc: intermediate files kept in {keep_dir}")

    if args.no_run:
        print("pylonc: --no-run given; not executing.")
        return 0

    # Step 4: run
    if embed:
        rc = run_exe(binary, args.args)
    else:
        rc = run_so(keep_dir, out_name, args.args)
    return rc


if __name__ == "__main__":
    sys.exit(main())
