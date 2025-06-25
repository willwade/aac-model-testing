"""
Test Cases Module

This module contains all the AAC-specific test cases for evaluating language models.
"""

from .base_test import BaseTest
from .text_correction_test import TextCorrectionTest
from .utterance_suggestion_test import UtteranceSuggestionTest
from .phraseboard_test import PhraseboardTest

__all__ = [
    "BaseTest",
    "TextCorrectionTest", 
    "UtteranceSuggestionTest",
    "PhraseboardTest"
]
