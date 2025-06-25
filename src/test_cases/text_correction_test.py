"""
Text Correction Test Case

This test evaluates a model's ability to correct poorly written sentences
that AAC users might produce, focusing on grammar, spelling, and clarity.
"""

from typing import Dict, Any, List
import re
from .base_test import BaseTest


class TextCorrectionTest(BaseTest):
    """
    Test case for evaluating text correction capabilities for AAC users.
    
    This test presents the model with grammatically incorrect or incomplete
    sentences that are typical of AAC user input and evaluates the model's
    ability to correct them into natural-sounding, grammatically correct sentences.
    """
    
    def __init__(self):
        super().__init__("Text Correction for AAC Users")
    
    def get_test_data(self) -> List[Dict[str, Any]]:
        """
        Get test data containing poorly written sentences and their corrections.
        
        Returns:
            List of test items with input sentences and expected corrections
        """
        return [
            {
                "id": "tc_001",
                "input": "me want eat pizza now hungry",
                "prompt": "Fix this sentence. Return only the corrected sentence, no explanations: {input}",
                "expected_corrections": [
                    "I want to eat pizza now because I'm hungry.",
                    "I want to eat pizza now. I'm hungry.",
                    "I want pizza now because I'm hungry."
                ],
                "difficulty": "easy",
                "error_types": ["missing_pronouns", "missing_articles", "missing_conjunctions"]
            },
            {
                "id": "tc_002",
                "input": "doctor appointment tomorrow need remember",
                "prompt": "Fix this sentence. Return only the corrected sentence, no explanations: {input}",
                "expected_corrections": [
                    "I have a doctor appointment tomorrow and need to remember.",
                    "I need to remember my doctor appointment tomorrow.",
                    "There's a doctor appointment tomorrow that I need to remember."
                ],
                "difficulty": "medium",
                "error_types": ["missing_pronouns", "missing_articles", "incomplete_thought"]
            },
            {
                "id": "tc_003",
                "input": "car broken can't go work today",
                "prompt": "Fix this sentence. Return only the corrected sentence, no explanations: {input}",
                "expected_corrections": [
                    "My car is broken, so I can't go to work today.",
                    "The car is broken and I can't go to work today.",
                    "My car broke down, so I can't go to work today."
                ],
                "difficulty": "medium",
                "error_types": ["missing_pronouns", "missing_articles", "missing_verbs"]
            },
            {
                "id": "tc_004",
                "input": "happy birthday mom love you much",
                "prompt": "Fix this sentence. Return only the corrected sentence, no explanations: {input}",
                "expected_corrections": [
                    "Happy birthday, Mom! I love you so much.",
                    "Happy birthday Mom, I love you very much.",
                    "Happy birthday to my mom, I love you so much."
                ],
                "difficulty": "easy",
                "error_types": ["missing_punctuation", "missing_pronouns", "missing_intensifiers"]
            },
            {
                "id": "tc_005",
                "input": "weather cold outside need jacket warm",
                "prompt": "Fix this sentence. Return only the corrected sentence, no explanations: {input}",
                "expected_corrections": [
                    "The weather is cold outside, so I need a warm jacket.",
                    "It's cold outside and I need a jacket to stay warm.",
                    "The weather is cold outside. I need a warm jacket."
                ],
                "difficulty": "medium",
                "error_types": ["missing_articles", "missing_verbs", "word_order"]
            },
            {
                "id": "tc_006",
                "input": "phone battery dead charge it please",
                "prompt": "Fix this sentence. Return only the corrected sentence, no explanations: {input}",
                "expected_corrections": [
                    "My phone battery is dead. Please charge it.",
                    "The phone battery is dead, can you charge it please?",
                    "My phone's battery died. Please charge it."
                ],
                "difficulty": "easy",
                "error_types": ["missing_pronouns", "missing_articles", "missing_verbs"]
            },
            {
                "id": "tc_007",
                "input": "store buy milk bread eggs list",
                "prompt": "Fix this sentence. Return only the corrected sentence, no explanations: {input}",
                "expected_corrections": [
                    "I need to go to the store to buy milk, bread, and eggs from my list.",
                    "Going to the store to buy milk, bread, and eggs on the list.",
                    "I'm going to the store to buy milk, bread, and eggs from the list."
                ],
                "difficulty": "hard",
                "error_types": ["missing_pronouns", "missing_articles", "missing_verbs", "incomplete_thought"]
            },
            {
                "id": "tc_008",
                "input": "tired sleep early tonight work tomorrow",
                "prompt": "Fix this sentence. Return only the corrected sentence, no explanations: {input}",
                "expected_corrections": [
                    "I'm tired, so I'll sleep early tonight because I have work tomorrow.",
                    "I'm tired and need to sleep early tonight since I work tomorrow.",
                    "I'm tired. I'll go to sleep early tonight because I have work tomorrow."
                ],
                "difficulty": "hard",
                "error_types": ["missing_pronouns", "missing_verbs", "missing_conjunctions"]
            },
            {
                "id": "tc_009",
                "input": "friend birthday party saturday fun time",
                "prompt": "Fix this sentence. Return only the corrected sentence, no explanations: {input}",
                "expected_corrections": [
                    "My friend's birthday party is on Saturday. It will be a fun time.",
                    "There's a friend's birthday party on Saturday, and it'll be fun.",
                    "My friend is having a birthday party on Saturday. It should be fun."
                ],
                "difficulty": "medium",
                "error_types": ["missing_pronouns", "missing_articles", "missing_verbs", "possessive"]
            },
            {
                "id": "tc_010",
                "input": "computer slow need fix help please",
                "prompt": "Fix this sentence. Return only the corrected sentence, no explanations: {input}",
                "expected_corrections": [
                    "My computer is slow and needs to be fixed. Please help.",
                    "The computer is running slow and needs fixing. Can you help please?",
                    "My computer is slow. I need help fixing it, please."
                ],
                "difficulty": "medium",
                "error_types": ["missing_pronouns", "missing_articles", "missing_verbs"]
            }
        ]
    
    def evaluate_response(self, test_item: Dict[str, Any], response: str) -> Dict[str, Any]:
        """
        Evaluate the model's text correction response.
        
        Args:
            test_item: The test item containing input and expected corrections
            response: The model's corrected sentence
        
        Returns:
            Dictionary containing evaluation metrics
        """
        # Clean the response
        cleaned_response = self._clean_response(response)
        
        # Initialize evaluation metrics
        evaluation = {
            "score": 0.0,
            "passed": False,
            "feedback": "",
            "metrics": {
                "grammar_score": 0.0,
                "completeness_score": 0.0,
                "naturalness_score": 0.0,
                "semantic_preservation": 0.0
            }
        }
        
        # Evaluate different aspects
        grammar_score = self._evaluate_grammar(test_item, cleaned_response)
        completeness_score = self._evaluate_completeness(test_item, cleaned_response)
        naturalness_score = self._evaluate_naturalness(test_item, cleaned_response)
        semantic_score = self._evaluate_semantic_preservation(test_item, cleaned_response)
        
        # Store individual metrics
        evaluation["metrics"]["grammar_score"] = grammar_score
        evaluation["metrics"]["completeness_score"] = completeness_score
        evaluation["metrics"]["naturalness_score"] = naturalness_score
        evaluation["metrics"]["semantic_preservation"] = semantic_score
        
        # Calculate overall score (weighted average)
        weights = {
            "grammar": 0.3,
            "completeness": 0.25,
            "naturalness": 0.25,
            "semantic": 0.2
        }
        
        overall_score = (
            grammar_score * weights["grammar"] +
            completeness_score * weights["completeness"] +
            naturalness_score * weights["naturalness"] +
            semantic_score * weights["semantic"]
        )
        
        evaluation["score"] = overall_score
        evaluation["passed"] = overall_score >= 0.7  # 70% threshold
        
        # Generate feedback
        evaluation["feedback"] = self._generate_feedback(
            test_item, cleaned_response, evaluation["metrics"]
        )
        
        return evaluation
    
    def _clean_response(self, response: str) -> str:
        """Clean and normalize the model's response."""
        # Remove thinking tags and their content
        cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)

        # Remove extra whitespace and normalize
        cleaned = re.sub(r'\s+', ' ', cleaned.strip())

        # Remove common prefixes that models might add
        prefixes_to_remove = [
            "Here's the corrected sentence:",
            "Here's a grammatically correct and natural-sounding version of your sentence:",
            "Here is the corrected sentence:",
            "Corrected:",
            "The corrected sentence is:",
            "Here is the correction:",
            "Fixed sentence:",
            "Corrected version:",
        ]

        for prefix in prefixes_to_remove:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix):].strip()

        # Remove markdown formatting
        cleaned = re.sub(r'\*\*(.*?)\*\*', r'\1', cleaned)  # Remove bold
        cleaned = re.sub(r'\*(.*?)\*', r'\1', cleaned)      # Remove italic

        # Remove quotes if the entire response is quoted
        if cleaned.startswith('"') and cleaned.endswith('"'):
            cleaned = cleaned[1:-1]

        # Extract just the first sentence if there are multiple sentences with explanations
        sentences = cleaned.split('.')
        if len(sentences) > 1:
            # Look for the actual corrected sentence (usually the first complete one)
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence.split()) >= 4:  # Reasonable sentence length
                    cleaned = sentence + '.'
                    break

        return cleaned
    
    def _evaluate_grammar(self, test_item: Dict[str, Any], response: str) -> float:
        """Evaluate grammatical correctness of the response."""
        # Basic grammar checks
        score = 1.0
        
        # Check for basic sentence structure
        if not response:
            return 0.0
        
        # Check for proper capitalization
        if not response[0].isupper():
            score -= 0.1
        
        # Check for ending punctuation
        if not response.rstrip().endswith(('.', '!', '?')):
            score -= 0.1
        
        # Check for common grammar patterns
        # This is a simplified check - in a real implementation,
        # you might use a grammar checking library
        
        # Check for subject-verb agreement patterns
        common_errors = [
            r'\bi\s+are\b',  # "I are"
            r'\bhe\s+are\b', # "he are"
            r'\bshe\s+are\b', # "she are"
        ]
        
        for pattern in common_errors:
            if re.search(pattern, response.lower()):
                score -= 0.2
        
        return max(0.0, score)
    
    def _evaluate_completeness(self, test_item: Dict[str, Any], response: str) -> float:
        """Evaluate if the response addresses all elements from the input."""
        input_words = set(test_item["input"].lower().split())
        response_words = set(response.lower().split())
        
        # Remove common stop words for better matching
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'}
        
        # Filter out stop words from input for key concept matching
        key_input_words = input_words - stop_words
        key_response_words = response_words - stop_words
        
        if not key_input_words:
            return 1.0
        
        # Calculate how many key concepts are preserved
        preserved_concepts = len(key_input_words.intersection(key_response_words))
        completeness_score = preserved_concepts / len(key_input_words)
        
        return min(1.0, completeness_score)
    
    def _evaluate_naturalness(self, test_item: Dict[str, Any], response: str) -> float:
        """Evaluate how natural and fluent the response sounds."""
        score = 1.0
        
        # Check for overly repetitive words
        words = response.lower().split()
        if len(words) > len(set(words)) * 1.5:  # Too much repetition
            score -= 0.2
        
        # Check for reasonable sentence length
        if len(words) < 3:
            score -= 0.3
        elif len(words) > 30:
            score -= 0.1
        
        # Check for natural word order and flow
        # This is simplified - could be enhanced with NLP libraries
        
        return max(0.0, score)
    
    def _evaluate_semantic_preservation(self, test_item: Dict[str, Any], response: str) -> float:
        """Evaluate if the core meaning is preserved."""
        # This is a simplified semantic evaluation
        # In a production system, you might use semantic similarity models
        
        test_item["input"].lower()
        response_text = response.lower()
        
        # Check if key semantic elements are present
        # This is a basic keyword-based approach
        score = 0.8  # Base score
        
        # Look for semantic consistency with expected corrections
        expected_corrections = test_item.get("expected_corrections", [])
        if expected_corrections:
            # Simple similarity check with expected corrections
            max_similarity = 0.0
            for expected in expected_corrections:
                similarity = self._calculate_simple_similarity(response_text, expected.lower())
                max_similarity = max(max_similarity, similarity)
            
            score = max_similarity
        
        return score
    
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
    
    def _generate_feedback(self, test_item: Dict[str, Any], response: str, metrics: Dict[str, float]) -> str:
        """Generate human-readable feedback for the response."""
        feedback_parts = []
        
        # Overall assessment
        overall_score = sum(metrics.values()) / len(metrics)
        if overall_score >= 0.8:
            feedback_parts.append("Excellent correction!")
        elif overall_score >= 0.6:
            feedback_parts.append("Good correction with room for improvement.")
        else:
            feedback_parts.append("Correction needs significant improvement.")
        
        # Specific feedback
        if metrics["grammar_score"] < 0.7:
            feedback_parts.append("Grammar could be improved.")
        
        if metrics["completeness_score"] < 0.7:
            feedback_parts.append("Some key information from the original text may be missing.")
        
        if metrics["naturalness_score"] < 0.7:
            feedback_parts.append("The sentence could sound more natural.")
        
        if metrics["semantic_preservation"] < 0.7:
            feedback_parts.append("The original meaning may not be fully preserved.")
        
        return " ".join(feedback_parts)
    
    def get_test_description(self) -> str:
        """Get description of this test case."""
        return ("Text Correction Test - Evaluates the model's ability to correct "
                "grammatically incorrect or incomplete sentences typical of AAC user input.")
    
    def get_scoring_criteria(self) -> Dict[str, str]:
        """Get scoring criteria for this test."""
        return {
            "grammar_score": "Grammatical correctness and proper sentence structure",
            "completeness_score": "Preservation of all key information from input",
            "naturalness_score": "How natural and fluent the corrected sentence sounds",
            "semantic_preservation": "How well the original meaning is maintained"
        }
