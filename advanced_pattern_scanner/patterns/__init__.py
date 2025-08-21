"""
Pattern detection algorithms for the Advanced Pattern Scanner.

This package contains implementations of various technical analysis patterns
based on established methodologies from authoritative trading sources.
"""

from .head_shoulders import HeadShouldersDetector
from .double_bottom import DoubleBottomDetector
from .cup_handle import CupHandleDetector

__all__ = [
    'HeadShouldersDetector',
    'DoubleBottomDetector', 
    'CupHandleDetector'
]