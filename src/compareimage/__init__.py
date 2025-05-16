"""
ImageDiff: A tool for comparing images and identifying content changes.

This package provides tools to detect, visualize, and report differences
between two images, with optional text detection and change grouping.
"""

__version__ = "0.1.0"

from compareimage.compare import compare_images, ZoomableChanges, display_results
from compareimage.report import generate_report, generate_text_report, generate_feature_differences

__all__ = [
    "compare_images",
    "ZoomableChanges",
    "display_results",
    "generate_report",
    "generate_text_report",
    "generate_feature_differences",
]