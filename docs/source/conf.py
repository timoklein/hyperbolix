# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Hyperbolic Math'
copyright = '2025, Thomas Lang, Kevin Sidak, Timo Klein'
author = 'Thomas Lang, Kevin Sidak, Timo Klein'
release = '2025'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

project = 'hyper_math'
copyright = '2025, Thomas Lang, Kevin Sidak'
author = 'Thomas Lang, Kevin Sidak'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx_autodoc_typehints',
]

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
