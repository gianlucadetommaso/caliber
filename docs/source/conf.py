import os
import sys

project = "caliber"
copyright = "2024, Gianluca Detommaso"
author = "Gianluca Detommaso"

sys.path.insert(0, os.path.abspath("../.."))
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..", "..")))


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",
    "sphinx.ext.mathjax",
    "sphinx_gallery.load_style",
    "sphinx.ext.viewcode",
]

napoleon_google_docstring = False

templates_path = ["_templates"]
exclude_patterns = []

autodoc_inherit_docstrings = True
autodoc_preserve_defaults = True
autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "undoc-members": True,
}
autoclass_content = "both"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_title = "caliber"
html_theme = "furo"
html_context = {
    "github_user": "gianlucadetommaso",
    "github_repo": "caliber",
    "github_version": "dev",
    "doc_path": "docs",
    "default_mode": "light",
}
htmlhelp_basename = "caliber"
html_show_sourcelink = False
