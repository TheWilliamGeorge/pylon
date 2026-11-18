#!/usr/bin/env python3
"""pylon2py.py - translate Pylon source code into pure Python.

Pylon is a Java-syntax variant of Python compiled via Cython (see
pylonc.py). This script does NOT compile to C; it translates the same
Pylon AST into idiomatic pure Python with PEP 484 type hints, preserving
the types the user wrote.

Usage:
    python pylon2py.py <file>.pylon [-o <file>.py]
    python pylon2py.py --stdout <file>.pylon        # print to stdout
    cat <file>.pylon | python pylon2py.py -          # read from stdin

The translator reuses pylonc.py's tokenizer, parser, AST, render_tokens
helper, and semantic checks (const-safety + return-type) so the two
translators can never drift in what they accept. The only thing this
script provides is a different Emitter that targets plain Python instead
of Cython.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

# Reuse the existing parser, AST, tokenizer, render_tokens, and the
# semantic checks. pylonc.py is expected to sit next to this file in
# scripts/ (or be importable via the same sys.path entry).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pylonc
from pylonc import (
    Token, Node, tokenize, Parser, render_tokens, PylonError,
    Decl, ConstDecl, FuncDef, ClassDef, IfStmt, WhileStmt, ForStmt,
    ReturnStmt, BreakStmt, ContinueStmt, PassStmt, ImportStmt,
    RaiseStmt, TryStmt, ExceptHandler, WithStmt, DelStmt, GlobalStmt,
    NonlocalStmt, AssertStmt, ExprStmt,
    _check_const_safety, _check_return_types,
)


# ---------------------------------------------------------------------------
# Trivia: comments and blank lines
# ---------------------------------------------------------------------------
#
# pylonc.py's lexer discards comments and newlines (it's a compiler; it
# only needs the AST). But pylon2py is a *converter*, so we want to
# preserve the author's comments and blank lines in the output. We do
# this with a separate scan of the raw source that finds comments and
# blank lines while respecting string literals (a `#` inside a string is
# not a comment). The result is a list of (line_no, kind, text) tuples
# that the emitter consults as it walks the AST.


@dataclass
class Trivia:
    """A comment or blank line in the original source."""
    line: int          # 1-based source line where this trivia lives
    kind: str          # "comment" | "blank"
    text: str = ""     # for comments: the full `# ...` text (no newline)


def extract_trivia(source: str) -> list[Trivia]:
    """Scan `source` for line comments and blank lines, skipping over
    string literals so a `#` inside a string isn't mistaken for a comment.

    A "blank line" is a line that is empty or contains only whitespace.
    Consecutive blank lines are collapsed: we emit one Trivia per blank
    line in the source so the emitter can reproduce the exact spacing.

    Triple-quoted strings can span multiple lines; we skip their contents
    so a `#` on a line inside a triple-quoted string is not treated as a
    comment.
    """
    trivia: list[Trivia] = []
    n = len(source)
    i = 0
    line = 1
    line_start = 0  # index of the first char of the current line

    def is_string_prefix(ch: str) -> bool:
        return ch in "fFrRbBuU"

    while i < n:
        c = source[i]

        # --- skip string literals (incl. prefixes, triple-quoted) ---------
        # We detect the start of a string and skip to its end, bumping `line`
        # for any newlines inside (triple-quoted). A `#` inside a string is
        # NOT a comment.
        starts_string = False
        prefix_end = i
        if c in "'\"":
            starts_string = True
            prefix_end = i
        else:
            # Look ahead for a string prefix (f, r, b, u, or combos like rb, fr)
            j = i
            while j < n and source[j] in "fFrRbBuU" and j - i < 2:
                j += 1
            if j > i and j < n and source[j] in "'\"":
                starts_string = True
                prefix_end = j
        if starts_string:
            quote = source[prefix_end]
            # Triple-quoted?
            triple = (prefix_end + 2 < n
                      and source[prefix_end + 1] == quote
                      and source[prefix_end + 2] == quote)
            if triple:
                end_q = prefix_end + 3
                while end_q < n:
                    if (source[end_q] == quote and end_q + 2 < n
                            and source[end_q + 1] == quote
                            and source[end_q + 2] == quote):
                        end_q += 3
                        break
                    if source[end_q] == "\n":
                        line += 1
                        line_start = end_q + 1
                    end_q += 1
                i = end_q
                continue
            else:
                # Single-line string; skip to closing quote (handle escapes).
                end_q = prefix_end + 1
                while end_q < n:
                    if source[end_q] == "\\" and end_q + 1 < n:
                        end_q += 2
                        continue
                    if source[end_q] == quote:
                        end_q += 1
                        break
                    if source[end_q] == "\n":
                        # Unterminated string; let the real lexer report it.
                        break
                    end_q += 1
                i = end_q
                continue

        # --- newline: check if the line we just finished was blank --------
        if c == "\n":
            line_content = source[line_start:i]
            if line_content.strip() == "":
                trivia.append(Trivia(line=line, kind="blank"))
            i += 1
            line += 1
            line_start = i
            continue

        # --- comment -------------------------------------------------------
        if c == "#":
            # Find end of line.
            end = i
            while end < n and source[end] != "\n":
                end += 1
            text = source[i:end].rstrip()
            trivia.append(Trivia(line=line, kind="comment", text=text))
            # Advance to the newline (the newline handler above will bump line).
            i = end
            continue

        i += 1

    return trivia


# ---------------------------------------------------------------------------
# Pylon type -> Python type hint
# ---------------------------------------------------------------------------

# Pylon type name -> Python type hint (as written in a PEP 484 annotation).
# `NoneType` means "returns nothing"; for a return type we emit `-> None`,
# for a variable annotation we emit `: None` (rare but legal).
PYLON_TO_PY_HINT = {
    "int":        "int",
    "float":      "float",
    "complex":    "complex",
    "boolean":    "bool",
    "bool":       "bool",
    "string":     "str",
    "str":        "str",
    "bytes":      "bytes",
    "bytearray":  "bytearray",
    "list":       "list",
    "tuple":      "tuple",
    "dict":       "dict",
    "set":        "set",
    "frozenset":  "frozenset",
    "range":      "range",
    "memoryview": "memoryview",
    "object":     "object",
    "NoneType":   "None",
}


def pylon_type_to_hint(name: Optional[str]) -> Optional[str]:
    """Map a Pylon type name to a Python type-hint string, or None if the
    Pylon type is unknown (we then silently drop the annotation rather
    than emit a broken hint)."""
    if name is None:
        return None
    return PYLON_TO_PY_HINT.get(name)


# ---------------------------------------------------------------------------
# Emitter: Pylon AST -> pure Python source
# ---------------------------------------------------------------------------

class PyEmitter:
    """Walk the Pylon AST and emit pure Python with PEP 484 type hints.

    Mirrors pylonc.Emitter's structure (emit/push/pop/render, emit_node
    dispatch, per-node emit_* methods) but targets plain Python:
      - `cdef int x = 5`      ->  `x: int = 5`
      - `cpdef int f(int n):`  ->  `def f(n: int) -> int:`
      - `const int MAX = 100` ->  `MAX: int = 100`   (const is a Pylon
                                 concept; Python has no const, so we
                                 preserve the type hint and let the
                                 const-safety check enforce immutability
                                 at compile time only)
      - Cython directives, cdef, cpdef, bint, long, etc. are all dropped.
    """

    def __init__(self, trivia: list[Trivia] = None) -> None:
        self.lines: list[str] = []
        self.indent = 0
        # True while we're inside a `class` body. Affects how typed
        # declarations are emitted (see emit_decl).
        self._in_class = 0
        # --- trivia (comments + blank lines) preservation ---
        # All trivia from the source, sorted by line. We pop from the
        # front as we advance past each line, emitting comments and
        # blank lines at the right indent before the next statement.
        self._trivia: list[Trivia] = sorted(
            trivia or [], key=lambda t: t.line)
        # The current source line we've emitted up to. Trivia with
        # `line <= self._cur_line` has already been flushed. Starts at 0
        # so trivia before the first statement (line 1+) is emitted.
        self._cur_line = 0
        # The source line of the last *statement* we emitted, used to
        # detect inline trailing comments (a `# ...` on the same source
        # line as the statement, e.g. `int x = 5;  # note`). None before
        # the first statement is emitted.
        self._last_stmt_line: Optional[int] = None

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

    # -- trivia (comments + blank lines) -----------------------------------
    def _flush_trivia_before(self, up_to_line: int) -> None:
        """Emit any comment or blank-line trivia whose source line is
        strictly less than `up_to_line` and hasn't been emitted yet.

        Called before emitting each statement node. The node's `.line`
        is passed as `up_to_line`; any trivia on lines above it (and not
        yet flushed) is emitted at the current indent level first. This
        preserves full-line comments, trailing comments on their own
        line, and blank-line spacing between statements.

        Trailing inline comments (a `# ...` on the SAME source line as
        the previous statement, e.g. `int x = 5;  # important`) are
        appended to the last emitted output line rather than starting a
        new line, so they stay attached to the statement they describe.

        Trivia that appears *after* the last statement (trailing
        comments at the end of a block or the module) is flushed by a
        separate call to `_flush_trivia_after` at the end of each body.
        """
        while self._trivia and self._trivia[0].line < up_to_line:
            t = self._trivia.pop(0)
            # Inline trailing comment: a `# ...` on the SAME source line
            # as the previous statement (e.g. `int x = 5;  # note`). This
            # must be checked BEFORE the `<= _cur_line` skip below,
            # because emitting the statement set `_cur_line` to that
            # statement's line, which equals this comment's line - the
            # skip would wrongly discard it. We append it to the last
            # emitted output line so it stays attached to its statement.
            if (t.kind == "comment"
                    and self._last_stmt_line is not None
                    and t.line == self._last_stmt_line
                    and self.lines):
                self.lines[-1] = self.lines[-1] + "  " + t.text
                self._cur_line = t.line
                continue
            # Skip trivia we've already accounted for. (After the
            # inline-comment check above, this only catches trivia on
            # lines strictly before the last statement - which is rare
            # but can happen with nested blocks.)
            if t.line <= self._cur_line:
                continue
            if t.kind == "blank":
                self.emit("")  # blank line at current indent (no text)
                self._cur_line = t.line
            else:  # comment (full-line or own-line)
                self.emit(t.text)  # `# ...` at current indent
                self._cur_line = t.line

    def _flush_trivia_after(self, up_to_line: int) -> None:
        """Emit trivia on lines >= up_to_line that belongs at the end of
        a body (trailing comments / blank lines after the last
        statement). Called when a block or module ends."""
        # Flush trivia whose line is between up_to_line and the next
        # statement's line (or EOF). We don't know the next statement's
        # line here, so we flush everything that's <= the next real
        # boundary. For module-end, flush all remaining trivia.
        while self._trivia and self._trivia[0].line >= up_to_line:
            t = self._trivia.pop(0)
            if t.kind == "blank":
                self.emit("")
            else:
                self.emit(t.text)
            self._cur_line = t.line

    # -- module top ---------------------------------------------------------
    def emit_module(self, prog: list[Node]) -> str:
        self.emit("# Generated by pylon2py from a .pylon source file.")
        self.emit("# Pure Python (PEP 484 type hints preserved from Pylon).")
        self.emit("# Run with: python <this file>.py")
        self.emit("")
        # NOTE: we deliberately do NOT set `_cur_line` to the header line
        # count (4). The header is generated content, not source content,
        # so it must not suppress source trivia. `_cur_line` starts at 0,
        # so trivia on source lines 1, 2, 3 ... (before the first
        # statement) is flushed normally.
        for node in prog:
            self._flush_trivia_before(node.line)
            self.emit_node(node)
            self._cur_line = node.line
            self._last_stmt_line = node.line
        # Flush any trailing trivia (comments/blank lines after the last
        # statement at module scope).
        self._flush_trivia_after(self._cur_line)
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
            raise PylonError(
                f"emit_node: unknown node type {type(node).__name__}"
            )

    # -- imports ------------------------------------------------------------
    def emit_import(self, node: ImportStmt) -> None:
        # Identical to pylonc.Emitter: Python import syntax is the same.
        if node.from_module is not None:
            parts = [f"from {node.from_module} import"]
            for name, alias in node.names:
                if alias:
                    parts.append(f"{name} as {alias}")
                else:
                    parts.append(name)
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
        hint = pylon_type_to_hint(node.type_name)
        # Inside a class body at class-definition time, PEP 484 variable
        # annotations without an assignment are valid (e.g. `x: int`), but
        # a bare `x: int` with no init is a class-level annotation, not
        # an instance attribute. To match the runtime behaviour Cython
        # gives for a typed module/global decl with no init (it zero-
        # inits C primitives, leaves objects as None), we emit `x: int
        # = None` for the no-init case so the name always exists.
        if node.init is not None:
            rhs = render_tokens(node.init)
            if hint is not None:
                self.emit(f"{node.name}: {hint} = {rhs}")
            else:
                self.emit(f"{node.name} = {rhs}")
        else:
            if hint is not None:
                self.emit(f"{node.name}: {hint} = None")
            else:
                self.emit(f"{node.name} = None")

    def emit_const(self, node: ConstDecl) -> None:
        # `const` is a Pylon concept; Python has no `const`. We preserve
        # the type hint and let the const-safety check enforce
        # immutability at Pylon-compile time. The const-safety check
        # already ran on the AST before this emitter saw it, so any
        # reassignment would have raised before we got here.
        hint = pylon_type_to_hint(node.type_name)
        rhs = render_tokens(node.init)
        if hint is not None:
            self.emit(f"{node.name}: {hint} = {rhs}")
        else:
            self.emit(f"{node.name} = {rhs}")

    # -- function -----------------------------------------------------------
    def emit_func(self, node: FuncDef) -> None:
        # `def ReturnType name (Type p1, Type p2) { body }`
        # ->  `def name(p1: Type, p2: Type) -> ReturnType:`
        #         body
        params_str = self._format_params(node.params)
        header = f"def {node.name}({params_str})"
        ret_hint = pylon_type_to_hint(node.return_type)
        if node.return_type is not None and ret_hint is not None:
            header += f" -> {ret_hint}"
        header += ":"
        self.emit(header)
        self.push()
        self._emit_body(node.body)
        self.pop()

    def _format_params(self, params: list[tuple[str, str]]) -> str:
        out: list[str] = []
        for ptype, pname in params:
            hint = pylon_type_to_hint(ptype)
            if hint is not None:
                out.append(f"{pname}: {hint}")
            else:
                out.append(pname)
        return ", ".join(out)

    # -- class --------------------------------------------------------------
    def emit_class(self, node: ClassDef) -> None:
        bases = ", ".join(node.bases)
        if bases:
            self.emit(f"class {node.name}({bases}):")
        else:
            self.emit(f"class {node.name}:")
        self.push()
        self._in_class += 1
        self._emit_body(node.body)
        self._in_class -= 1
        self.pop()

    # -- control flow -------------------------------------------------------
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
        """Emit a list of statements, flushing comment/blank-line trivia
        before each one so the author's comments and spacing survive the
        translation. An empty body emits `pass` (Python requires a body)."""
        if not body:
            self.emit("pass")
            return
        for s in body:
            # Flush any comments/blank lines that appear above this
            # statement in the source, at the current indent level.
            self._flush_trivia_before(s.line)
            self.emit_node(s)
            self._cur_line = s.line
            self._last_stmt_line = s.line
        # Trailing comments/blank lines after the last statement in this
        # body are flushed by the enclosing construct (try/finally not
        # needed because the enclosing emit_* calls _flush_trivia_after
        # via the next outer body or module end). We deliberately do NOT
        # flush remaining trivia here, because the next statement in the
        # *outer* scope owns those lines.


# ---------------------------------------------------------------------------
# Translation entry point
# ---------------------------------------------------------------------------

def translate_to_python(source: str) -> str:
    """Translate Pylon source code into pure Python source code.

    Runs the same tokenizer, parser, and semantic checks as pylonc.py
    (so the two translators never disagree on what's valid), then emits
    pure Python via PyEmitter. Comments and blank lines from the source
    are preserved (pylon2py is a converter, not just a compiler).
    """
    tokens = tokenize(source)
    parser = Parser(tokens)
    program = parser.parse_program()
    # Reuse pylonc's semantic checks verbatim - they operate on the AST
    # and are language-agnostic (they don't care whether the target is
    # Cython or Python).
    _check_const_safety(program)
    _check_return_types(program)
    # Extract comments + blank lines (trivia) from the raw source, so the
    # emitter can re-insert them at the right indent in the output.
    trivia = extract_trivia(source)
    emitter = PyEmitter(trivia=trivia)
    return emitter.emit_module(program)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pylon2py",
        description="Translate Pylon source code into pure Python "
                    "(with PEP 484 type hints).",
    )
    p.add_argument("source",
                   help="Path to a .pylon source file, or '-' for stdin.")
    p.add_argument("-o", "--output", default=None,
                   help="Write Python to this path. If omitted, the "
                        "output goes next to the source file with a .py "
                        "extension (or to stdout for stdin input).")
    p.add_argument("--stdout", action="store_true",
                   help="Print the Python to stdout instead of writing "
                        "a file (overrides -o).")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    # Read source.
    if args.source == "-":
        source = sys.stdin.read()
        src_path: Optional[str] = None
    else:
        src_path = args.source
        if not os.path.isfile(src_path):
            sys.stderr.write(f"pylon2py: no such file: {src_path}\n")
            return 2
        with open(src_path, "r", encoding="utf-8") as f:
            source = f.read()

    # Translate.
    try:
        python_source = translate_to_python(source)
    except PylonError as e:
        sys.stderr.write(f"pylon2py: translation error: {e}\n")
        return 1

    # Write output.
    if args.stdout:
        sys.stdout.write(python_source)
        return 0

    if args.output:
        out_path = args.output
    elif src_path is None:
        # stdin input with no -o and no --stdout: print to stdout.
        sys.stdout.write(python_source)
        return 0
    else:
        base, _ = os.path.splitext(src_path)
        out_path = base + ".py"

    out_dir = os.path.dirname(os.path.abspath(out_path)) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(python_source)
    sys.stderr.write(f"pylon2py: wrote {out_path}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
