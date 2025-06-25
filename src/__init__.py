"""
AAC Model Testing Framework

A comprehensive testing framework to evaluate small offline language models 
for Augmentative and Alternative Communication (AAC) use cases.
"""

__version__ = "1.0.0"
__author__ = "AAC Model Testing Framework"

from .aac_testing_framework import AACTestingFramework
from .model_manager import ModelManager, ModelWrapper
from .performance_monitor import PerformanceMonitor
from .results_analyzer import ResultsAnalyzer

__all__ = [
    "AACTestingFramework",
    "ModelManager", 
    "ModelWrapper",
    "PerformanceMonitor",
    "ResultsAnalyzer"
]
