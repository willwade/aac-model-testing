"""
Pydantic schemas for structured model outputs.

This module defines the data structures that models should return
for each type of AAC test case, ensuring consistent and parseable responses.
"""

from typing import List
from pydantic import BaseModel, Field


class TextCorrectionResponse(BaseModel):
    """Schema for text correction responses."""
    corrected_text: str = Field(
        description="The grammatically corrected version of the input text",
        min_length=1
    )


class UtteranceSuggestion(BaseModel):
    """A single utterance suggestion."""
    text: str = Field(
        description="The suggested utterance/sentence",
        min_length=3
    )
    category: str = Field(
        description="Category of the utterance (request, question, statement, etc.)",
        default="statement"
    )


class UtteranceSuggestionsResponse(BaseModel):
    """Schema for utterance suggestion responses."""
    suggestions: List[UtteranceSuggestion] = Field(
        description="List of 4 different utterance suggestions",
        min_length=4,
        max_length=4
    )


class PhraseItem(BaseModel):
    """A single phrase or word for a phrase board."""
    text: str = Field(
        description="The phrase or word",
        min_length=1,
        max_length=50
    )
    category: str = Field(
        description="Category of the phrase (action, description, object, etc.)",
        default="general"
    )


class PhraseboardResponse(BaseModel):
    """Schema for phraseboard generation responses."""
    phrases: List[PhraseItem] = Field(
        description="List of 12 words/phrases for the topic",
        min_length=12,
        max_length=12
    )
    topic: str = Field(
        description="The topic these phrases relate to"
    )
