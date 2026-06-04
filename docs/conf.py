import sys, os
sys.path.insert(0, os.path.abspath("../src"))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "pelagos-py"
copyright = "2025, National Oceanography Centre"
author = "National Oceanography Centre"

# Single-source the version from setuptools-scm (i.e. the latest git tag).
# ``release`` is the full version (e.g. "2.4.1.dev3+gabc1234"); ``version`` is
# the short X.Y form shown in the docs. Both feed the |release| / |version|
# substitutions used on the index page.
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _get_version

try:
    release = _get_version("pelagos_py")
except PackageNotFoundError:
    # Package not installed in the build env; read the version straight from
    # the git tree so local `make html` still shows a real number.
    try:
        from setuptools_scm import get_version

        release = get_version(root="..", relative_to=__file__)
    except Exception:
        release = "0.0.0+unknown"

version = ".".join(release.split(".")[:2])

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

autoapi_dirs = ["../src"]
autoapi_root = "api"

autoapi_keep_files = False

# Visibility is driven entirely by whether a member has a docstring. With
# "undoc-members" removed, anything without a docstring is hidden automatically
# (e.g. ``run``, ``step_name``, ``parameter_schema``), giving the clean,
# scipy-like pages where the class docstring is the description and only the
# methods you choose to document (e.g. ``generate_diagnostics``) appear.
autoapi_options = [
    "members",  # Include members (methods, attributes)...
    "show-inheritance",  # ...and show the base classes.
    # Deliberately NOT enabled: "undoc-members" (hides docstring-less noise),
    # "special-members" (hides __init__ etc.), "imported-members" (hides
    # re-imported names like xr/pd/np/BaseStep), "show-module-summary" (the
    # custom template below renders members directly, no summary table).
]

# Use our trimmed module template (docs/_templates/autoapi/python/module.rst):
# module pages show just the title + the documented class, dropping the module
# docstring, the summary table, and the "Module Contents" heading.
autoapi_template_dir = "_templates/autoapi"


# AutoAPI parses source *statically* (it never imports your package), so a
# runtime marker set by a decorator -- e.g. ``obj.__nodoc__ = True`` -- is
# invisible at build time. The supported escape hatch is the skip event below:
# put ``:meta private:`` anywhere in a docstring to force-hide that object even
# though it is documented.
def _autoapi_skip_member(app, what, name, obj, skip, options):
    if getattr(obj, "docstring", "") and ":meta private:" in obj.docstring:
        return True
    return skip


def setup(app):
    app.connect("autoapi-skip-member", _autoapi_skip_member)

# autoapi_ignore = [
#     "pelagos_pypy.metadata_parser.metadata_parser",  # Exclude it as a submodule
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
html_title = "Pelagos-Py"
html_favicon = "_static/favicon.svg"
html_css_files = ["custom.css"]

# Pages with no sub-navigation — remove empty left sidebar
html_sidebars = {
    "getting_started": [],
    "development": [],
}

html_theme_options = {
    "logo": {
        "image_light": "_static/pelagos_logo.png",
        "image_dark": "_static/pelagos_logo.png",
        "text": "Pelagos-Py",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/NOC-OBG-Autonomy/pelagos-py",
            "icon": "fa-brands fa-github",
        }
    ],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "show_prev_next": False,
}
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
