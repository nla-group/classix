# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------
import sys
import os
import sphinx_rtd_theme


sys.path.insert(0, os.path.abspath('../..'))

project = 'CLASSIX'
copyright = '2022, Stefan Güttel, Xinye Chen'
author = 'Stefan Güttel, Xinye Chen'

# The full version, including alpha/beta/rc tags
release = '0.0.3'


# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx_rtd_theme',
]
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
source_suffix = '.rst'
exclude_patterns = []
locale_dirs = ['locale/']
gettext_compact = False

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

pygments_style = 'lovelace'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_style = 'css/_.css'
html_theme = "sphinx_rtd_theme" # html_theme = 'alabaster'
html_logo = "classix_logo.png"
html_theme_options = {
    'logo_only': False,
    'navigation_depth': 5,
}
