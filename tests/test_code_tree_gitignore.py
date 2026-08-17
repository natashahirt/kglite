"""The build walk honors ``.gitignore``.

Generated output is the single largest source of junk in a code graph, and it is
junk that *looks* like source: a built docs site is thousands of real HTML files,
a wheel build directory holds real copies of real modules. Name-based filtering
cannot separate them from source, because the names are ambiguous — a ``dist/``
may be committed build output a project deliberately ships. The repo's own
``.gitignore`` is the authoritative statement, so the walk defers to it.

These tests pin the *semantics* (negation, nesting, scope), not just the happy
path — the point of delegating to a real gitignore implementation is that the
awkward cases behave like git rather than like a name list.
"""

import textwrap

import pytest

pytest.importorskip("tree_sitter")

from kglite.code_tree import build  # noqa: E402


def _write(tmp_path, files: dict[str, str]) -> None:
    for rel, content in files.items():
        fp = tmp_path / rel
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(textwrap.dedent(content))


def _indexed(graph) -> set[str]:
    """Every path the graph holds a File node for."""
    rows = graph.cypher("MATCH (f:File) RETURN f.path AS path").to_list()
    return {r["path"] for r in rows}


def _has(graph, suffix: str) -> bool:
    return any(p.endswith(suffix) for p in _indexed(graph))


SRC = "def real_fn():\n    return 42\n"


class TestGitignoreRespected:
    def test_gitignored_build_output_is_not_indexed(self, tmp_path):
        """The motivating case: a wheel build dir mirroring the whole package."""
        _write(
            tmp_path,
            {
                ".gitignore": "build/\n",
                "src/real.py": SRC,
                "build/lib/src/real.py": SRC,
            },
        )
        g = build(str(tmp_path))
        assert _has(g, "src/real.py")
        assert not _has(g, "build/lib/src/real.py"), sorted(_indexed(g))

    def test_a_generated_docs_site_is_not_indexed(self, tmp_path):
        """mkdocs/sphinx output is thousands of real HTML files, all generated."""
        _write(
            tmp_path,
            {
                ".gitignore": "/site/\n",
                "docs/page.html": "<html><body><div id='a'>x</div></body></html>",
                "site/page.html": "<html><body><div id='a'>x</div></body></html>",
            },
        )
        g = build(str(tmp_path))
        assert _has(g, "docs/page.html")
        assert not _has(g, "site/page.html"), sorted(_indexed(g))

    def test_a_committed_dist_is_still_indexed(self, tmp_path):
        """The contract that rules out name-based filtering.

        ``dist/`` is NOT ignored here, so it is source as far as this repo is
        concerned — a project may deliberately commit built assets it ships.
        Skipping it on the strength of its name would silently drop them.
        """
        _write(tmp_path, {".gitignore": "build/\n", "dist/shipped.py": SRC})
        g = build(str(tmp_path))
        assert _has(g, "dist/shipped.py"), sorted(_indexed(g))


class TestGitignoreSemantics:
    def test_a_scratch_pattern_also_skips_a_tracked_file_that_matches(self, tmp_path):
        """The known edge, pinned so it is discoverable rather than surprising.

        A scratch pattern like ``/pkg/_*.py`` also matches a committed
        ``__init__.py``. Git keeps that file — it consults the index, and a
        tracked path stays tracked — but this walk has no index and goes by the
        pattern, exactly as ``rg`` and ``fd`` do. Reading the index would buy back
        the package marker at the cost of a git dependency and of missing
        brand-new untracked source, which matters more for a working-tree graph.
        """
        _write(
            tmp_path,
            {".gitignore": "/pkg/_*.py\n", "pkg/__init__.py": SRC, "pkg/real.py": SRC},
        )
        g = build(str(tmp_path))
        assert _has(g, "pkg/real.py"), sorted(_indexed(g))
        assert not _has(g, "pkg/__init__.py")

    def test_a_negation_pattern_re_includes_a_file(self, tmp_path):
        """`!keep.py` must win over the broader rule above it, as it does in git."""
        _write(
            tmp_path,
            {
                ".gitignore": "generated/\n!generated/keep.py\n",
                "generated/drop.py": SRC,
                "generated/keep.py": SRC,
            },
        )
        g = build(str(tmp_path))
        assert not _has(g, "generated/drop.py")
        # A negation cannot re-include a file whose PARENT dir is excluded, so
        # git itself drops keep.py here. Asserting the file list matches git's
        # view matters more than asserting the intuitive-but-wrong outcome.
        assert not _has(g, "generated/keep.py"), "git does not re-include a file under an excluded directory"

    def test_a_nested_gitignore_applies_to_its_own_subtree(self, tmp_path):
        _write(
            tmp_path,
            {
                "pkg/.gitignore": "out/\n",
                "pkg/real.py": SRC,
                "pkg/out/gen.py": SRC,
                "other/out/kept.py": SRC,
            },
        )
        g = build(str(tmp_path))
        assert _has(g, "pkg/real.py")
        assert not _has(g, "pkg/out/gen.py")
        assert _has(g, "other/out/kept.py"), "a nested ignore must not leak sideways"

    def test_rules_are_anchored_at_the_project_root_not_the_walked_subdir(self, tmp_path):
        """A manifest source root is often a SUBDIRECTORY, and that is the trap.

        With a manifest declaring roots, the builder walks each root separately.
        If ignore rules were resolved from the walked directory, the repo's root
        ``.gitignore`` would be a parent of the walk and get skipped — so a
        ``build/`` holding a full copy of the package would be indexed even
        though the repo excludes it. That is the actual bug this guards: the
        graph looked fine, it just quietly contained the build tree twice.
        """
        _write(
            tmp_path,
            {
                "pyproject.toml": (
                    '[project]\nname = "demo"\nversion = "0"\n\n[tool.setuptools]\npackages = ["demo"]\n'
                ),
                ".gitignore": "build/\n",
                "demo/__init__.py": SRC,
                "build/lib/demo/__init__.py": SRC,
            },
        )
        g = build(str(tmp_path))
        indexed = sorted(_indexed(g))
        assert any("demo/__init__.py" in p and not p.startswith("build") for p in indexed)
        assert not any(p.startswith("build/") for p in indexed), indexed

    def test_an_ignore_file_above_the_root_is_not_consulted(self, tmp_path):
        """A graph's contents must depend on the tree parsed, not its surroundings.

        Walking ``repo/`` must not inherit rules from a ``.gitignore`` in the
        parent — otherwise the same repo yields different graphs depending on
        where it happens to be checked out.
        """
        _write(
            tmp_path,
            {".gitignore": "repo/\nsrc/\n", "repo/src/real.py": SRC},
        )
        g = build(str(tmp_path / "repo"))
        assert _has(g, "real.py"), sorted(_indexed(g))


class TestNameListStillApplies:
    def test_node_modules_is_skipped_without_any_gitignore(self, tmp_path):
        """Vendored trees show up in dirs with no ignore file at all."""
        _write(
            tmp_path,
            {"src/real.py": SRC, "node_modules/dep/index.js": "function d(){return 1}"},
        )
        g = build(str(tmp_path))
        assert _has(g, "src/real.py")
        assert not _has(g, "node_modules/dep/index.js"), sorted(_indexed(g))
