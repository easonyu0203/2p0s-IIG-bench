"""Sphinx configuration for the nashbench documentation."""

project = "nashbench"
author = "Eason Yu"
copyright = "2026, Eason Yu"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]
myst_enable_extensions = ["dollarmath"]
myst_heading_anchors = 2
autodoc_member_order = "bysource"
exclude_patterns = ["_build"]

html_theme = "sphinx_book_theme"
html_title = "nashbench"
html_theme_options = {
    "repository_url": "https://github.com/easonyu0203/2p0s-IIG-bench",
    "use_repository_button": True,
}
