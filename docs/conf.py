# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import re
import sys
from pathlib import Path

# -- Path setup ---------------------------------------------------------------
# Allow autodoc to find the package source without an install.
DOCS = Path(__file__).resolve().parent
REPO = DOCS.parent
sys.path.insert(0, str(REPO / "src"))

# -- Project information ------------------------------------------------------
project = "zvDVC"
copyright = "2026, Andrew Keenlyside. zvDVC reimplements the DVC method of iDVC and the CCPi DVC code (B. K. Bay et al.)"
author = "Andrew Keenlyside"
try:
    from importlib.metadata import version as _pkg_version

    release = _pkg_version("zvdvc")
except Exception:  # pragma: no cover - docs build without an install
    release = "0.0.0+unknown"
version = release.split("+")[0]

GITHUB = "https://github.com/Andrew-Keenlyside/zvDVC"
GITHUB_BRANCH = "main"

# -- General configuration ----------------------------------------------------
extensions = [
    # Core
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    # Markdown support
    "myst_parser",
    # Copy button on code blocks
    "sphinx_copybutton",
    # CLI reference from zvdvc.cli.build_parser
    "sphinxarg.ext",
]

# MyST-Parser configuration
myst_enable_extensions = [
    "colon_fence",      # ::: directive syntax
    "deflist",          # definition lists
    "fieldlist",        # field lists
    "dollarmath",       # $...$ and $$...$$ maths
    "attrs_inline",     # inline attribute syntax
]
myst_heading_anchors = 3

# Napoleon (Google / NumPy docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# autodoc
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_member_order = "bysource"
# GPU, JIT and optional-extra modules are imported lazily inside functions, but mock them
# so a docs build (Read the Docs has no GPU) never tries to load them.
autodoc_mock_imports = ["cupy", "cupyx", "numba", "kvikio", "mpi4py", "tifffile"]

# intersphinx — link to upstream docs
intersphinx_mapping = {
    "python":       ("https://docs.python.org/3", None),
    "numpy":        ("https://numpy.org/doc/stable", None),
    "zarr":         ("https://zarr.readthedocs.io/en/stable", None),
    "zarr_vectors": ("https://zarr-vectors-py.readthedocs.io/en/latest", None),
}

# Source suffixes
source_suffix = {
    ".rst": "restructuredtext",
    ".md":  "markdown",
}
master_doc = "index"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "assets"]
suppress_warnings = ["myst.header"]   # the design notes and dated reports start below h1 in places

# -- HTML output --------------------------------------------------------------
html_theme = "furo"
html_title = "zvDVC"

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary":    "#e0195c",
        "color-brand-content":    "#e0195c",
        "font-stack":             "'DM Sans', sans-serif",
        "font-stack--monospace":  "'JetBrains Mono', monospace",
    },
    "dark_css_variables": {
        "color-brand-primary":    "#ff72c0",
        "color-brand-content":    "#ff72c0",
    },
    "sidebar_hide_name": True,     # the logo carries the name
    "navigation_with_keys": True,
    "top_of_page_button": "edit",
    "source_repository": f"{GITHUB}/",
    "source_branch": GITHUB_BRANCH,
    "source_directory": "docs/",
}

html_logo = "_static/zvdvc-logo.png"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# Show "Edit on GitHub" links
html_context = {
    "github_user":    "Andrew-Keenlyside",
    "github_repo":    "zvDVC",
    "github_version": GITHUB_BRANCH,
    "doc_path":       "docs",
}

# -- copybutton ---------------------------------------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True


# -- Links out of docs/ -------------------------------------------------------
# The design notes and benchmark reports are also read on GitHub, so they link to
# source files with relative paths (``../src/zvdvc/io/volume.py``). Sphinx cannot
# resolve a path outside docs/, so point those links at the file on GitHub instead.
_LINK = re.compile(r"\]\((?!https?:|mailto:|#|/)([^)\s#]+)(#[^)\s]*)?\)")


def _rewrite_links_out_of_docs(app, docname, source):
    here = (DOCS / docname).parent

    def repl(m):
        target, anchor = m.group(1), m.group(2) or ""
        path = (here / target).resolve()
        try:
            path.relative_to(DOCS)
            return m.group(0)                    # inside docs/: leave it to Sphinx
        except ValueError:
            pass
        try:
            rel = path.relative_to(REPO).as_posix()
        except ValueError:
            return m.group(0)
        kind = "tree" if path.is_dir() else "blob"
        return f"]({GITHUB}/{kind}/{GITHUB_BRANCH}/{rel}{anchor})"

    source[0] = _LINK.sub(repl, source[0])


def setup(app):
    app.connect("source-read", _rewrite_links_out_of_docs)
