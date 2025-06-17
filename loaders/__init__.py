"""
Data loading components for various sources.
"""

from .web_loader import WebDataLoader
from .file_loader import FileDataLoader

__all__ = ['WebDataLoader', 'FileDataLoader']
