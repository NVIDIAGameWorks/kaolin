# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
sys.path.append('../')
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'kaolin')))
# import sphinx_rtd_theme

# The master toctree document.
master_doc = 'index'


# -- Project information -----------------------------------------------------

project = 'kaolin'
copyright = '2019, NVIDIA Development Inc.'
author = 'NVIDIA'

# The full version, including alpha/beta/rc tags
release = '0.1.0 alpha'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
]

napoleon_use_ivar = True

# Mock CUDA Imports
autodoc_mock_imports = ['kaolin.cuda.ball_query',
                        'kaolin.cuda.load_textures',
                        'kaolin.cuda.sided_distance',
                        'kaolin.cuda.furthest_point_sampling',
                        'kaolin.cuda.three_nn',
                        'kaolin.cuda.tri_distance',
                        'kaolin.cuda.mesh_intersection',
                        'kaolin.graphics.nmr.cuda.rasterize_cuda']


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# The name of the Pygments (syntax highlighting) style to use
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing
todo_include_todos = True

# Do not prepend module name to functions
add_module_names = False


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'
html_theme_path = ['_themes']

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_css_files = [
    'copybutton.css',
]
html_js_files = [
    'clipboard.min.js',
    'copybutton.js',
]


# -- Options for LaTeX output -------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper')
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt')
    #
    # 'pointsize': '10pt',

    # Font packages
    'fontpkg': '\\usepackage{amsmath, amsfonts, amssymb, amsthm}'

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LateX files. List of tuples
# (source start file, target name, title,
# author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'kaolin.tex', u'kaolin Documentation',
        [author], 1),
]


# -- Options for manual page output -------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'kaolin', u'kaolin Documentation', 
        [author], 1)
]


# -- Options for Texinfo output -----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author, 
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'kaolin', 'kaolin Documentation', 
        author, 'kaolin', 'NVIDIA 3D Deep Learning Library.', 
        'Miscellaneous'),
]


# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('http://docs.scipy.org/doc/numpy', None),
    'torch': ('http://pytorch.org/docs/master', None),
}
