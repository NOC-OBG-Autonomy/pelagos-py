# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "pelagos-py"
copyright = "2026, National Oceanography Centre"
author = "Adam Ward & Daniel Bangay, National Oceanography Centre"
version = "0.0.0.1"
release = version

# -- General configuration ------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named "sphinx.ext.*") or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    "nbsphinx",
    "numpydoc",
    "autoapi.extension",
    "myst_parser",
    "sphinx_last_updated_by_git",
    "sphinx_codeautolink",
    "sphinx_design",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# -- Napoleon configuration for NumPy-style docstrings -------------------

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_param = True
napoleon_use_rtype = True

# -- Use autoapi.extension to run sphinx-apidoc -------

autoapi_dirs = ["../src/toolbox"]
autoapi_root = "api"

autoapi_keep_files = False
autoapi_options = [
    "members",  # Include all members (methods, attributes)
    "undoc-members",  # Include undocumented members
    "show-inheritance",  # Show class inheritance
    "show-module-summary",  # Show module summary
    "special-members",  # Include special members (e.g., __init__)
    "imported-members",  # Include imported members (if applicable)
    "no-private-members",  # Exclude private members (optional)
]

# autoapi_ignore = [
#     "toolboxpy.metadata_parser.metadata_parser",  # Exclude it as a submodule
# ]

# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = "sphinx_rtd_theme"
html_theme = "pydata_sphinx_theme"

html_baseurl = "https://noc-obg-autonomy.github.io/pelagos-py/"
html_static_path = ["_static"]
html_last_updated_fmt = "%b %d, %Y"
html_show_sourcelink = False
# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#


nbsphinx_execute = "always"
nbsphinx_allow_errors = True
nbsphinx_kernel_name = "python3"
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
]

nbsphinx_thumbnails = {
    "gallery/thumbnail-from-conf-py": "gallery/a-local-file.png",
    "gallery/*-rst": "images/notebook_icon.png",
    "orphan": "_static/favicon.svg",
}

# -- Options for Intersphinx

intersphinx_mapping = {
    "IPython": ("https://ipython.readthedocs.io/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/", None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "python": ("https://docs.python.org/3/", None),
}

autodoc_default_options = {
    "member-order": "bysource",
    "undoc-members": True,
    "show-inheritance": True,
    "fullqualname": True,
}

add_module_names = True

pygments_style = "friendly"
pygments_dark_style = "monokai"

copybutton_prompt_text = r">>> |\$ "
copybutton_prompt_is_regexp = True


myst_enable_extensions = [
    "colon_fence",  # ::: fenced blocks
    "deflist",  # definition lists
    "linkify",  # auto-detect bare links
    "smartquotes",  # nicer quotes/dashes
]

# Optional: create anchors for h1-h3 automatically
myst_heading_anchors = 3