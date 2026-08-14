#!/usr/bin/env python3
"""On-demand conformance check for `File -[:IMPORTS]-> File` edges.

Rebuilds the import graph of a target repository with Python's own ``ast`` module
and diffs it against the one kglite's tree-sitter parsers produce. The oracle
shares none of the machinery under test — not the AST walk, not the module-path
rooting, not the resolver — so a disagreement localises a real resolution
difference rather than a difference of opinion about which files to scan: the file
universe is taken from kglite's own File nodes.

Like the other conformance scripts here, this is deliberately **not** part of the
pytest suite. It needs a large real repository to be meaningful, and the
per-language breakdown it prints is for reading, not asserting.

Usage:

    python scripts/import_graph_conformance.py /path/to/some/repo

Reports recall and precision against the oracle, lists disagreements in both
directions, and breaks edges down by language pair so cross-language resolution
(always a bug) is visible at a glance.
"""

import ast
from pathlib import Path
import sys

REPO = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()


def module_names(rel: str) -> str:
    """The dotted module a repo-relative .py path is imported by."""
    parts = rel[:-3].split("/") if rel.endswith(".py") else rel.split("/")
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def oracle(files: set[str]) -> set[tuple[str, str]]:
    # Sorted iteration plus explicit package precedence, so a module path claimed
    # by both `x/__init__.py` and `x.py` resolves the same way on every run
    # (matching Python's own precedence) instead of following set order.
    mod_to_file = {}
    for rel in sorted(files):
        name = module_names(rel)
        if not name:
            continue
        if name not in mod_to_file or rel.endswith("__init__.py"):
            mod_to_file[name] = rel

    edges: set[tuple[str, str]] = set()
    for rel in sorted(files):
        try:
            src = (REPO / rel).read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(src)
        except (OSError, SyntaxError):
            continue

        own = module_names(rel)
        is_pkg = rel.endswith("__init__.py")
        pkg = own if is_pkg else own.rpartition(".")[0]

        targets: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                targets += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = pkg.split(".")
                    for _ in range(1, node.level):
                        if base:
                            base.pop()
                    resolved = ".".join(base)
                    if node.module:
                        resolved = f"{resolved}.{node.module}" if resolved else node.module
                    if resolved:
                        targets.append(resolved)
                        # `from .pkg import name` may name a submodule, not an attr.
                        targets += [f"{resolved}.{a.name}" for a in node.names]
                elif node.module:
                    targets.append(node.module)
                    targets += [f"{node.module}.{a.name}" for a in node.names]

        for t in targets:
            parts = t.split(".")
            for end in range(len(parts), 0, -1):
                hit = mod_to_file.get(".".join(parts[:end]))
                if hit:
                    if hit != rel:
                        edges.add((rel, hit))
                    break
    return edges


def main() -> None:
    from kglite._kglite_code_tree import build

    g = build(str(REPO), verbose=False)

    def rows(q):
        return [dict(r) for r in g.cypher(q, params={})]

    py_files = {r["p"] for r in rows("MATCH (f:File) WHERE f.language = 'python' RETURN f.path AS p")}
    kgl = {(r["a"], r["b"]) for r in rows("MATCH (a:File)-[:IMPORTS]->(b:File) RETURN a.path AS a, b.path AS b")}
    kgl_py = {(a, b) for a, b in kgl if a in py_files and b in py_files}
    orc = oracle(py_files)

    print(f"python files scanned      : {len(py_files)}")
    print(f"oracle edges (ast)        : {len(orc)}")
    print(f"kglite edges (python)     : {len(kgl_py)}")
    print(f"kglite edges (all langs)  : {len(kgl)}")
    hit = len(kgl_py & orc)
    print(f"agreement                 : {hit}")
    print(f"  recall  (oracle found)  : {hit / len(orc):.1%}" if orc else "")
    print(f"  precision (kglite right): {hit / len(kgl_py):.1%}" if kgl_py else "")

    missing = sorted(orc - kgl_py)
    extra = sorted(kgl_py - orc)
    print(f"\nmissing from kglite: {len(missing)}")
    for a, b in missing[:15]:
        print(f"  {a} -> {b}")
    print(f"\nkglite-only (not in oracle): {len(extra)}")
    for a, b in extra[:15]:
        print(f"  {a} -> {b}")

    mods = rows("MATCH (f:File)-[:IMPORTS]->(m:Module) RETURN DISTINCT m.qualified_name AS q")
    print(f"\ndistinct File->Module targets: {len(mods)}")

    print("\n--- per-language file->file edges ---")
    lang = {r["p"]: r["l"] for r in rows("MATCH (f:File) RETURN f.path AS p, f.language AS l")}
    by_pair: dict[tuple[str, str], int] = {}
    for a, b in kgl:
        key = (lang.get(a) or "?", lang.get(b) or "?")
        by_pair[key] = by_pair.get(key, 0) + 1
    for (la, lb), n in sorted(by_pair.items(), key=lambda kv: -kv[1]):
        flag = "" if la == lb or {la, lb} <= {"javascript", "typescript"} else "   <-- CROSS-LANGUAGE"
        print(f"  {la:12} -> {lb:12} {n:6}{flag}")


if __name__ == "__main__":
    main()
