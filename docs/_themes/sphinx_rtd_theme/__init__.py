"""
Sphinx Read the Docs theme.

From https://github.com/ryan-roemer/sphinx-bootstrap-theme.
"""

from os import path

import sphinx


__version__ = '0.4.3.dev0'
__version_full__ = __version__


def get_html_theme_path():
    """Return list of HTML theme paths."""
    cur_dir = path.abspath(path.dirname(path.dirname(__file__)))
    return cur_dir

def scb_static_path(app):
    static_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'static'))
    app.config.html_static_path.append(static_path)

clipboard_js_url = "https://cdnjs.cloudflare.com/ajax/libs/clipboard.js/2.0.0/clipboard.min.js"


# See http://www.sphinx-doc.org/en/stable/theming.html#distribute-your-theme-as-a-python-package
def setup(app):
    app.add_html_theme('sphinx_rtd_theme', path.abspath(path.dirname(__file__)))
    app.connect('builder-inited', scb_static_path)
    app.add_stylesheet('static/copybutton.css')
    app.add_javascript('static/clipboard.min.js')
    app.add_javascript('static/copybutton.js')

    if sphinx.version_info >= (1, 8, 0):
        # Add Sphinx message catalog for newer versions of Sphinx
        # See http://www.sphinx-doc.org/en/master/extdev/appapi.html#sphinx.application.Sphinx.add_message_catalog
        rtd_locale_path = path.join(path.abspath(path.dirname(__file__)), 'locale')
        app.add_message_catalog('sphinx', rtd_locale_path)
