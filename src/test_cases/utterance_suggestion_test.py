"""
Utterance Suggestion Test Case

This test evaluates a model's ability to generate multiple phrase suggestions
from minimal input, helping AAC users express their thoughts more completely.
"""

from typing import Dict, Any, List
import re
from .base_test import BaseTest


class UtteranceSuggestionTest(BaseTest):
    """
    Test case for evaluating utterance suggestion generation for AAC users.
    
    This test presents the model with minimal input (like keywords) and evaluates
    the model's ability to generate multiple relevant, complete phrases that
    an AAC user might want to express.
    """
    
    def __init__(self):
        super().__init__("Utterance Suggestion Generation")
    
    def get_test_data(self) -> List[Dict[str, Any]]:
        """
        Get test data containing minimal inputs and expected phrase suggestions.
        
        Returns:
            List of test items with keyword inputs and expected phrase categories
        """
        return [
            {
                "id": "us_001",
                "input": "pizza vegetarian",
                "prompt": "Generate 4 phrases from these keywords: {input}. Return only the phrases separated by commas, no explanations: phrase1, phrase2, phrase3, phrase4",
                "expected_categories": ["request", "question", "preference", "statement"],
                "expected_phrases": [
                    "I want a vegetarian pizza",
                    "Do you have vegetarian pizza?",
                    "I love vegetarian pizza",
                    "Can I order a vegetarian pizza?",
                    "I prefer vegetarian pizza"
                ],
                "context": "food_ordering",
                "difficulty": "easy"
            },
            {
                "id": "us_002",
                "input": "doctor appointment",
                "prompt": "Generate 4 phrases from these keywords: {input}. Return only the phrases separated by commas, no explanations: phrase1, phrase2, phrase3, phrase4",
                "expected_categories": ["scheduling", "reminder", "question", "statement"],
                "expected_phrases": [
                    "I have a doctor appointment",
                    "I need to schedule a doctor appointment",
                    "When is my doctor appointment?",
                    "I forgot about my doctor appointment",
                    "Can you remind me about my doctor appointment?"
                ],
                "context": "healthcare",
                "difficulty": "medium"
            },
            {
                "id": "us_003",
                "input": "tired sleep",
                "prompt": "Generate 4 phrases from these keywords: {input}. Return only the phrases separated by commas, no explanations: phrase1, phrase2, phrase3, phrase4",
                "expected_categories": ["statement", "request", "explanation"],
                "expected_phrases": [
                    "I'm tired and need to sleep",
                    "I'm too tired to sleep",
                    "I want to go to sleep because I'm tired",
                    "I'm tired, can I sleep now?",
                    "I'm feeling tired and sleepy"
                ],
                "context": "personal_state",
                "difficulty": "easy"
            },
            {
                "id": "us_004",
                "input": "help computer",
                "prompt": "Generate 4 phrases from these keywords: {input}. Return only the phrases separated by commas, no explanations: phrase1, phrase2, phrase3, phrase4",
                "expected_categories": ["request", "question", "problem"],
                "expected_phrases": [
                    "I need help with my computer",
                    "Can you help me with the computer?",
                    "My computer needs help",
                    "Help me fix my computer",
                    "I'm having computer problems, can you help?"
                ],
                "context": "technology_support",
                "difficulty": "medium"
            },
            {
                "id": "us_005",
                "input": "birthday party friend",
                "prompt": "Generate 4 phrases from these keywords: {input}. Return only the phrases separated by commas, no explanations: phrase1, phrase2, phrase3, phrase4",
                "expected_categories": ["invitation", "statement", "question", "planning"],
                "expected_phrases": [
                    "My friend is having a birthday party",
                    "I'm going to my friend's birthday party",
                    "When is your friend's birthday party?",
                    "I want to plan a birthday party for my friend",
                    "Are you coming to my friend's birthday party?"
                ],
                "context": "social_events",
                "difficulty": "medium"
            },
            {
                "id": "us_006",
                "input": "weather cold jacket",
                "prompt": "Generate 4 phrases from these keywords: {input}. Return only the phrases separated by commas, no explanations: phrase1, phrase2, phrase3, phrase4",
                "expected_categories": ["observation", "request", "statement"],
                "expected_phrases": [
                    "The weather is cold, I need a jacket",
                    "It's cold outside, where's my jacket?",
                    "I need my jacket because it's cold",
                    "The cold weather requires a jacket",
                    "Can you get my jacket? It's cold"
                ],
                "context": "weather_clothing",
                "difficulty": "easy"
            },
            {
                "id": "us_007",
                "input": "store buy milk",
                "prompt": "Generate 4 phrases from these keywords: {input}. Return only the phrases separated by commas, no explanations: phrase1, phrase2, phrase3, phrase4",
                "expected_categories": ["task", "reminder", "request", "statement"],
                "expected_phrases": [
                    "I need to go to the store to buy milk",
                    "Don't forget to buy milk at the store",
                    "Can you buy milk from the store?",
                    "I'm going to the store for milk",
                    "We need milk from the store"
                ],
                "context": "shopping",
                "difficulty": "easy"
            },
            {
                "id": "us_008",
                "input": "phone call mom",
                "prompt": "Generate 4 phrases from these keywords: {input}. Return only the phrases separated by commas, no explanations: phrase1, phrase2, phrase3, phrase4",
                "expected_categories": ["task", "reminder", "statement", "request"],
                "expected_phrases": [
                    "I need to call my mom",
                    "Mom called me on the phone",
                    "Can you help me call mom?",
                    "I want to make a phone call to mom",
                    "Remind me to call mom"
                ],
                "context": "communication",
                "difficulty": "easy"
            },
            {
                "id": "us_009",
                "input": "work meeting important",
                "prompt": "Generate 4 phrases from these keywords: {input}. Return only the phrases separated by commas, no explanations: phrase1, phrase2, phrase3, phrase4",
                "expected_categories": ["statement", "reminder", "question", "concern"],
                "expected_phrases": [
                    "I have an important work meeting",
                    "The work meeting is very important",
                    "When is the important meeting at work?",
                    "I can't miss this important work meeting",
                    "This work meeting is important to me"
                ],
                "context": "workplace",
                "difficulty": "medium"
            },
            {
                "id": "us_010",
                "input": "happy excited vacation",
                "prompt": "Generate 4 phrases from these keywords: {input}. Return only the phrases separated by commas, no explanations: phrase1, phrase2, phrase3, phrase4",
                "expected_categories": ["emotion", "statement", "sharing"],
                "expected_phrases": [
                    "I'm happy and excited about my vacation",
                    "I feel excited and happy about vacation",
                    "My vacation makes me happy and excited",
                    "I'm so happy and excited for vacation",
                    "Vacation time makes me happy and excited"
                ],
                "context": "emotions_travel",
                "difficulty": "easy"
            }
        ]
    
    def evaluate_response(self, test_item: Dict[str, Any], response: str) -> Dict[str, Any]:
        """
        Evaluate the model's utterance suggestion response.
        
        Args:
            test_item: The test item containing input and expected suggestions
            response: The model's generated phrase suggestions
        
        Returns:
            Dictionary containing evaluation metrics
        """
        # Parse the response into individual phrases
        phrases = self._parse_phrases(response)
        
        # Initialize evaluation metrics
        evaluation = {
            "score": 0.0,
            "passed": False,
            "feedback": "",
            "metrics": {
                "phrase_count": len(phrases),
                "completeness_score": 0.0,
                "relevance_score": 0.0,
                "diversity_score": 0.0,
                "quality_score": 0.0
            }
        }
        
        if not phrases:
            evaluation["feedback"] = "No valid phrases were generated."
            return evaluation
        
        # Evaluate different aspects
        completeness_score = self._evaluate_completeness(phrases)
        relevance_score = self._evaluate_relevance(test_item, phrases)
        diversity_score = self._evaluate_diversity(phrases)
        quality_score = self._evaluate_quality(phrases)
        
        # Store individual metrics
        evaluation["metrics"]["completeness_score"] = completeness_score
        evaluation["metrics"]["relevance_score"] = relevance_score
        evaluation["metrics"]["diversity_score"] = diversity_score
        evaluation["metrics"]["quality_score"] = quality_score
        
        # Calculate overall score (weighted average)
        weights = {
            "completeness": 0.2,
            "relevance": 0.35,
            "diversity": 0.25,
            "quality": 0.2
        }
        
        overall_score = (
            completeness_score * weights["completeness"] +
            relevance_score * weights["relevance"] +
            diversity_score * weights["diversity"] +
            quality_score * weights["quality"]
        )
        
        evaluation["score"] = overall_score
        evaluation["passed"] = overall_score >= 0.7 and len(phrases) >= 3
        
        # Generate feedback
        evaluation["feedback"] = self._generate_feedback(test_item, phrases, evaluation["metrics"])
        
        return evaluation
    
    def _parse_phrases(self, response: str) -> List[str]:
        """Parse the response into individual phrases from CSV format."""
        # Clean the response first
        cleaned = self._clean_response(response)

        # Primary strategy: Split by commas (CSV format)
        phrases = []

        # Split by commas and clean each phrase
        comma_split = [phrase.strip() for phrase in cleaned.split(',') if phrase.strip()]

        if len(comma_split) > 1:
            # CSV format detected
            phrases = comma_split
        else:
            # Fallback: Try other splitting strategies for backward compatibility
            # Strategy 1: Split by newlines
            lines = [line.strip() for line in cleaned.split('\n') if line.strip()]

            # Strategy 2: If no newlines, try numbered lists
            if len(lines) == 1:
                # Look for numbered patterns like "1.", "2.", etc.
                numbered_pattern = r'\d+\.\s*'
                if re.search(numbered_pattern, cleaned):
                    parts = re.split(numbered_pattern, cleaned)
                    lines = [part.strip() for part in parts if part.strip()]

            # Strategy 3: If still one line, try bullet points
            if len(lines) == 1:
                bullet_pattern = r'[•\-\*]\s*'
                if re.search(bullet_pattern, cleaned):
                    parts = re.split(bullet_pattern, cleaned)
                    lines = [part.strip() for part in parts if part.strip()]

            phrases = lines

        # Clean each phrase
        cleaned_phrases = []
        for phrase in phrases:
            # Remove common prefixes
            prefixes_to_remove = [
                "Here are some suggestions:",
                "Possible phrases:",
                "Suggestions:",
                "-",
                "•",
                "*"
            ]

            cleaned_phrase = phrase
            for prefix in prefixes_to_remove:
                if cleaned_phrase.startswith(prefix):
                    cleaned_phrase = cleaned_phrase[len(prefix):].strip()

            # Remove quotes
            if cleaned_phrase.startswith('"') and cleaned_phrase.endswith('"'):
                cleaned_phrase = cleaned_phrase[1:-1]

            # Only include if it's a reasonable phrase (3+ words)
            if len(cleaned_phrase.split()) >= 3:
                cleaned_phrases.append(cleaned_phrase)

        return cleaned_phrases

    def _clean_response(self, response: str) -> str:
        """Clean and normalize the model's response."""
        # Remove thinking tags and their content
        cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)

        # Remove extra whitespace and normalize
        cleaned = re.sub(r'\s+', ' ', cleaned.strip())

        # Remove common prefixes that models might add
        prefixes_to_remove = [
            "Here are the phrases:",
            "Here are some phrases:",
            "Phrases:",
            "Generated phrases:",
            "The phrases are:",
            "Here's the list:",
            "Here are 4 phrases:",
            "4 phrases:",
        ]

        for prefix in prefixes_to_remove:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix):].strip()

        # Remove markdown formatting
        cleaned = re.sub(r'\*\*(.*?)\*\*', r'\1', cleaned)  # Remove bold
        cleaned = re.sub(r'\*(.*?)\*', r'\1', cleaned)      # Remove italic

        # Remove bullet points at the start
        cleaned = re.sub(r'^[-•*]\s*', '', cleaned)

        # Remove quotes if the entire response is quoted
        if cleaned.startswith('"') and cleaned.endswith('"'):
            cleaned = cleaned[1:-1]

        return cleaned

    def _evaluate_completeness(self, phrases: List[str]) -> float:
        """Evaluate if the response contains an appropriate number of phrases."""
        phrase_count = len(phrases)
        
        # Ideal range is 3-5 phrases
        if 3 <= phrase_count <= 5:
            return 1.0
        elif phrase_count == 2:
            return 0.7
        elif phrase_count == 6:
            return 0.9
        elif phrase_count == 1:
            return 0.4
        elif phrase_count == 0:
            return 0.0
        else:  # 7+ phrases
            return max(0.5, 1.0 - (phrase_count - 5) * 0.1)
    
    def _evaluate_relevance(self, test_item: Dict[str, Any], phrases: List[str]) -> float:
        """Evaluate how relevant the phrases are to the input keywords."""
        input_keywords = test_item["input"].lower().split()
        total_relevance = 0.0
        
        for phrase in phrases:
            phrase_words = set(phrase.lower().split())
            
            # Check keyword inclusion
            keyword_matches = sum(1 for keyword in input_keywords if keyword in phrase_words)
            keyword_score = keyword_matches / len(input_keywords)
            
            # Check semantic relevance (simplified)
            semantic_score = self._calculate_semantic_relevance(test_item, phrase)
            
            # Combine scores
            phrase_relevance = (keyword_score * 0.6) + (semantic_score * 0.4)
            total_relevance += phrase_relevance
        
        return total_relevance / len(phrases) if phrases else 0.0
    
    def _calculate_semantic_relevance(self, test_item: Dict[str, Any], phrase: str) -> float:
        """Calculate semantic relevance of a phrase to the expected context."""
        # This is a simplified approach - in production, you might use
        # semantic similarity models or word embeddings
        
        expected_phrases = test_item.get("expected_phrases", [])
        if not expected_phrases:
            return 0.5  # Neutral score if no expected phrases
        
        phrase_lower = phrase.lower()
        max_similarity = 0.0
        
        for expected in expected_phrases:
            similarity = self._calculate_simple_similarity(phrase_lower, expected.lower())
            max_similarity = max(max_similarity, similarity)
        
        return max_similarity
    
    def _calculate_simple_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple word-based similarity between two texts."""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _evaluate_diversity(self, phrases: List[str]) -> float:
        """Evaluate the diversity of the generated phrases."""
        if len(phrases) <= 1:
            return 0.0
        
        # Check lexical diversity
        all_words = []
        for phrase in phrases:
            all_words.extend(phrase.lower().split())
        
        unique_words = set(all_words)
        lexical_diversity = len(unique_words) / len(all_words) if all_words else 0
        
        # Check structural diversity (different sentence patterns)
        patterns = set()
        for phrase in phrases:
            # Simple pattern detection based on first few words
            words = phrase.split()
            if len(words) >= 2:
                pattern = f"{words[0].lower()}_{words[1].lower()}"
                patterns.add(pattern)
        
        structural_diversity = len(patterns) / len(phrases)
        
        # Combine diversity scores
        return (lexical_diversity * 0.6) + (structural_diversity * 0.4)
    
    def _evaluate_quality(self, phrases: List[str]) -> float:
        """Evaluate the overall quality of the generated phrases."""
        if not phrases:
            return 0.0
        
        total_quality = 0.0
        
        for phrase in phrases:
            phrase_quality = 1.0
            
            # Check for proper capitalization
            if not phrase[0].isupper():
                phrase_quality -= 0.1
            
            # Check for reasonable length
            word_count = len(phrase.split())
            if word_count < 3:
                phrase_quality -= 0.3
            elif word_count > 15:
                phrase_quality -= 0.1
            
            # Check for completeness (should be a complete thought)
            if not self._is_complete_sentence(phrase):
                phrase_quality -= 0.2
            
            total_quality += max(0.0, phrase_quality)
        
        return total_quality / len(phrases)
    
    def _is_complete_sentence(self, phrase: str) -> bool:
        """Check if a phrase represents a complete sentence/thought."""
        # Simple heuristics for completeness
        words = phrase.lower().split()
        
        # Should have a subject and predicate
        has_pronoun = any(word in words for word in ['i', 'you', 'he', 'she', 'it', 'we', 'they', 'my', 'your'])
        has_verb = any(word in words for word in ['am', 'is', 'are', 'was', 'were', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'can', 'could', 'should', 'need', 'want', 'like', 'love', 'go', 'come', 'get', 'make', 'take', 'give'])
        
        return has_pronoun and has_verb
    
    def _generate_feedback(self, test_item: Dict[str, Any], phrases: List[str], metrics: Dict[str, float]) -> str:
        """Generate human-readable feedback for the response."""
        feedback_parts = []
        
        # Overall assessment
        phrase_count = len(phrases)
        if phrase_count == 0:
            return "No valid phrases were generated. Please provide 3-5 complete phrases."
        
        if phrase_count < 3:
            feedback_parts.append(f"Only {phrase_count} phrase(s) generated. Aim for 3-5 phrases.")
        elif phrase_count > 5:
            feedback_parts.append(f"{phrase_count} phrases generated. 3-5 is ideal for AAC users.")
        else:
            feedback_parts.append(f"Good number of phrases ({phrase_count}).")
        
        # Quality feedback
        if metrics["quality_score"] >= 0.8:
            feedback_parts.append("Phrases are well-formed and complete.")
        elif metrics["quality_score"] >= 0.6:
            feedback_parts.append("Most phrases are good quality.")
        else:
            feedback_parts.append("Phrase quality could be improved.")
        
        # Relevance feedback
        if metrics["relevance_score"] >= 0.8:
            feedback_parts.append("Phrases are highly relevant to the input.")
        elif metrics["relevance_score"] >= 0.6:
            feedback_parts.append("Phrases are mostly relevant.")
        else:
            feedback_parts.append("Phrases could be more relevant to the input keywords.")
        
        # Diversity feedback
        if metrics["diversity_score"] >= 0.7:
            feedback_parts.append("Good variety in phrase structures.")
        else:
            feedback_parts.append("Could use more diverse phrase patterns.")
        
        return " ".join(feedback_parts)
    
    def get_test_description(self) -> str:
        """Get description of this test case."""
        return ("Utterance Suggestion Test - Evaluates the model's ability to generate "
                "multiple relevant phrase suggestions from minimal keyword input.")
    
    def get_scoring_criteria(self) -> Dict[str, str]:
        """Get scoring criteria for this test."""
        return {
            "completeness_score": "Appropriate number of phrases generated (3-5 ideal)",
            "relevance_score": "How well phrases relate to input keywords and context",
            "diversity_score": "Variety in phrase structures and vocabulary",
            "quality_score": "Overall quality, completeness, and grammatical correctness"
        }
