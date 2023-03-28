"""Coherence Mixture Model"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("cmm")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Virginia M.S. Rutten"
__email__ = "vms.rutten@gmail.com"
