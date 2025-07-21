"""
learnit::r8 - An Intelligent ML Training Scheduler
===================================================

This package provides the R8Scheduler, a tool for managing 
the ML training process with advanced features like data-aware
batching, dynamic LR scheduling, and live UI monitoring.
"""

__version__ = "1.0.0"
__author__ = "Gemini AI"

from .scheduler import R8Scheduler

# This makes `from learnit_r8 import R8Scheduler` possible.
__all__ = ['R8Scheduler']