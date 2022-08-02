"""
.. include:: ../../README.md
"""

import logging

import pkg_resources

# Fetches the version of the package as defined in pyproject.toml
__version__ = pkg_resources.get_distribution("dr_hatespeech").version

# Set up logging
fmt = "%(asctime)s [%(levelname)s] <%(name)s> %(message)s"
logging.basicConfig(level=logging.INFO, format=fmt)
