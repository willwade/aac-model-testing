"""
Phrase Board Test Case

This test evaluates a model's ability to generate 12 relevant words or phrases
for a given topic that would be useful for AAC phrase boards.
"""

from typing import Dict, Any, List
import re
import csv
from io import StringIO
from .base_test import BaseTest


class PhraseboardTest(BaseTest):
    """
    Test case for evaluating topic-based phrase board generation for AAC users.
    
    This test presents the model with a topic and evaluates the model's ability
    to generate 12 relevant words or phrases that would be useful for an AAC
    phrase board on that topic, returned in CSV format.
    """
    
    def __init__(self):
        super().__init__("Topic-Based Phrase Board Generation")
    
    def get_test_data(self) -> List[Dict[str, Any]]:
        """
        Get test data containing topics and expected phrase categories.
        
        Returns:
            List of test items with topics and expected phrase types
        """
        return [
            {
                "id": "pb_001",
                "input": "dogs",
                "prompt": "For this user suggest words or phrases based on the input text of: {input} that would be useful to have as a phrase board based on this topic. Return as a CSV with exactly 12 items, one per line.",
                "expected_categories": ["actions", "descriptions", "needs", "emotions", "objects"],
                "expected_phrases": [
                    "walk the dog", "feed the dog", "pet the dog", "play with dog",
                    "good dog", "cute dog", "big dog", "small dog",
                    "dog food", "dog toy", "leash", "collar"
                ],
                "topic_type": "pets",
                "difficulty": "easy"
            },
            {
                "id": "pb_002",
                "input": "cooking",
                "prompt": "For this user suggest words or phrases based on the input text of: {input} that would be useful to have as a phrase board based on this topic. Return as a CSV with exactly 12 items, one per line.",
                "expected_categories": ["actions", "tools", "ingredients", "methods", "results"],
                "expected_phrases": [
                    "cook dinner", "bake cookies", "chop vegetables", "stir the pot",
                    "knife", "pan", "oven", "spoon",
                    "salt", "pepper", "oil", "flour"
                ],
                "topic_type": "activities",
                "difficulty": "medium"
            },
            {
                "id": "pb_003",
                "input": "school",
                "prompt": "For this user suggest words or phrases based on the input text of: {input} that would be useful to have as a phrase board based on this topic. Return as a CSV with exactly 12 items, one per line.",
                "expected_categories": ["subjects", "people", "objects", "actions", "places"],
                "expected_phrases": [
                    "math class", "reading time", "science project", "art class",
                    "teacher", "student", "principal", "friend",
                    "book", "pencil", "computer", "homework"
                ],
                "topic_type": "education",
                "difficulty": "medium"
            },
            {
                "id": "pb_004",
                "input": "weather",
                "prompt": "For this user suggest words or phrases based on the input text of: {input} that would be useful to have as a phrase board based on this topic. Return as a CSV with exactly 12 items, one per line.",
                "expected_categories": ["conditions", "temperature", "activities", "clothing", "feelings"],
                "expected_phrases": [
                    "sunny day", "rainy weather", "cloudy sky", "windy outside",
                    "hot", "cold", "warm", "cool",
                    "umbrella", "jacket", "sunglasses", "boots"
                ],
                "topic_type": "environment",
                "difficulty": "easy"
            },
            {
                "id": "pb_005",
                "input": "birthday party",
                "prompt": "For this user suggest words or phrases based on the input text of: {input} that would be useful to have as a phrase board based on this topic. Return as a CSV with exactly 12 items, one per line.",
                "expected_categories": ["activities", "objects", "people", "food", "emotions"],
                "expected_phrases": [
                    "blow out candles", "open presents", "sing happy birthday", "play games",
                    "cake", "balloons", "presents", "party hat",
                    "friends", "family", "birthday person", "guests"
                ],
                "topic_type": "celebrations",
                "difficulty": "medium"
            },
            {
                "id": "pb_006",
                "input": "hospital",
                "prompt": "For this user suggest words or phrases based on the input text of: {input} that would be useful to have as a phrase board based on this topic. Return as a CSV with exactly 12 items, one per line.",
                "expected_categories": ["people", "feelings", "needs", "objects", "actions"],
                "expected_phrases": [
                    "doctor", "nurse", "patient", "visitor",
                    "scared", "worried", "better", "sick",
                    "help me", "I need", "it hurts", "thank you",
                    "medicine", "bed", "wheelchair", "bandage"
                ],
                "topic_type": "healthcare",
                "difficulty": "hard"
            },
            {
                "id": "pb_007",
                "input": "shopping",
                "prompt": "For this user suggest words or phrases based on the input text of: {input} that would be useful to have as a phrase board based on this topic. Return as a CSV with exactly 12 items, one per line.",
                "expected_categories": ["actions", "objects", "places", "needs", "payment"],
                "expected_phrases": [
                    "buy groceries", "find item", "check price", "pay now",
                    "shopping cart", "shopping list", "receipt", "bag",
                    "store", "checkout", "aisle", "cashier"
                ],
                "topic_type": "activities",
                "difficulty": "medium"
            },
            {
                "id": "pb_008",
                "input": "transportation",
                "prompt": "For this user suggest words or phrases based on the input text of: {input} that would be useful to have as a phrase board based on this topic. Return as a CSV with exactly 12 items, one per line.",
                "expected_categories": ["vehicles", "actions", "places", "needs", "safety"],
                "expected_phrases": [
                    "car", "bus", "train", "airplane",
                    "drive", "ride", "walk", "travel",
                    "station", "airport", "stop", "destination",
                    "ticket", "seatbelt", "safe trip", "arrive"
                ],
                "topic_type": "travel",
                "difficulty": "medium"
            }
        ]
    
    def evaluate_response(self, test_item: Dict[str, Any], response: str) -> Dict[str, Any]:
        """
        Evaluate the model's phrase board generation response.
        
        Args:
            test_item: The test item containing topic and expected categories
            response: The model's generated CSV phrase list
        
        Returns:
            Dictionary containing evaluation metrics
        """
        # Parse the CSV response
        phrases = self._parse_csv_response(response)
        
        # Initialize evaluation metrics
        evaluation = {
            "score": 0.0,
            "passed": False,
            "feedback": "",
            "metrics": {
                "phrase_count": len(phrases),
                "format_score": 0.0,
                "relevance_score": 0.0,
                "diversity_score": 0.0,
                "aac_suitability_score": 0.0
            }
        }
        
        if not phrases:
            evaluation["feedback"] = "No valid phrases were parsed from the response."
            return evaluation
        
        # Evaluate different aspects
        format_score = self._evaluate_format(phrases, response)
        relevance_score = self._evaluate_relevance(test_item, phrases)
        diversity_score = self._evaluate_diversity(test_item, phrases)
        aac_suitability_score = self._evaluate_aac_suitability(phrases)
        
        # Store individual metrics
        evaluation["metrics"]["format_score"] = format_score
        evaluation["metrics"]["relevance_score"] = relevance_score
        evaluation["metrics"]["diversity_score"] = diversity_score
        evaluation["metrics"]["aac_suitability_score"] = aac_suitability_score
        
        # Calculate overall score (weighted average)
        weights = {
            "format": 0.2,
            "relevance": 0.35,
            "diversity": 0.25,
            "aac_suitability": 0.2
        }
        
        overall_score = (
            format_score * weights["format"] +
            relevance_score * weights["relevance"] +
            diversity_score * weights["diversity"] +
            aac_suitability_score * weights["aac_suitability"]
        )
        
        evaluation["score"] = overall_score
        evaluation["passed"] = overall_score >= 0.7 and len(phrases) >= 10  # Allow some flexibility
        
        # Generate feedback
        evaluation["feedback"] = self._generate_feedback(test_item, phrases, evaluation["metrics"])
        
        return evaluation
    
    def _parse_csv_response(self, response: str) -> List[str]:
        """Parse the CSV response into individual phrases."""
        phrases = []
        
        # Clean the response
        cleaned = response.strip()
        
        # Remove common prefixes
        prefixes_to_remove = [
            "Here are 12 words/phrases:",
            "Here's a CSV list:",
            "CSV format:",
            "Here are the phrases:",
        ]
        
        for prefix in prefixes_to_remove:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix):].strip()
        
        # Try to parse as CSV
        try:
            # Use StringIO to treat the string as a file
            csv_reader = csv.reader(StringIO(cleaned))
            for row in csv_reader:
                if row:  # Skip empty rows
                    # Take the first column if multiple columns
                    phrase = row[0].strip()
                    if phrase:
                        phrases.append(phrase)
        except:
            # If CSV parsing fails, try line-by-line parsing
            lines = [line.strip() for line in cleaned.split('\n') if line.strip()]
            for line in lines:
                # Remove common list markers
                line = re.sub(r'^\d+\.\s*', '', line)  # Remove "1. "
                line = re.sub(r'^[•\-\*]\s*', '', line)  # Remove bullet points
                line = line.strip('"\'')  # Remove quotes
                
                if line and len(line.split()) <= 5:  # Reasonable phrase length
                    phrases.append(line)
        
        return phrases[:12]  # Limit to 12 phrases as requested
    
    def _evaluate_format(self, phrases: List[str], original_response: str) -> float:
        """Evaluate if the response follows the requested CSV format."""
        score = 1.0
        
        # Check if exactly 12 phrases
        if len(phrases) == 12:
            score += 0.0  # Perfect
        elif 10 <= len(phrases) <= 14:
            score -= 0.1  # Close enough
        elif 8 <= len(phrases) <= 16:
            score -= 0.3  # Acceptable
        else:
            score -= 0.5  # Poor count
        
        # Check if response looks like CSV format
        if ',' in original_response or '\n' in original_response:
            score += 0.0  # Good format
        else:
            score -= 0.2  # Poor format
        
        return max(0.0, score)
    
    def _evaluate_relevance(self, test_item: Dict[str, Any], phrases: List[str]) -> float:
        """Evaluate how relevant the phrases are to the topic."""
        topic = test_item["input"].lower()
        topic_words = set(topic.split())
        
        relevant_count = 0
        total_relevance = 0.0
        
        for phrase in phrases:
            phrase_words = set(phrase.lower().split())
            
            # Direct topic word inclusion
            direct_match = len(topic_words.intersection(phrase_words)) > 0
            
            # Semantic relevance (simplified)
            semantic_relevance = self._calculate_topic_relevance(test_item, phrase)
            
            # Combine relevance scores
            phrase_relevance = 0.3 * (1.0 if direct_match else 0.0) + 0.7 * semantic_relevance
            
            if phrase_relevance >= 0.5:
                relevant_count += 1
            
            total_relevance += phrase_relevance
        
        return total_relevance / len(phrases) if phrases else 0.0
    
    def _calculate_topic_relevance(self, test_item: Dict[str, Any], phrase: str) -> float:
        """Calculate how relevant a phrase is to the topic context."""
        # This is a simplified approach using expected phrases
        expected_phrases = test_item.get("expected_phrases", [])
        
        if not expected_phrases:
            return 0.5  # Neutral if no expected phrases
        
        phrase_lower = phrase.lower()
        max_similarity = 0.0
        
        # Check similarity with expected phrases
        for expected in expected_phrases:
            similarity = self._calculate_simple_similarity(phrase_lower, expected.lower())
            max_similarity = max(max_similarity, similarity)
        
        # Also check for topic-related keywords
        topic_keywords = self._get_topic_keywords(test_item["input"])
        keyword_matches = sum(1 for keyword in topic_keywords if keyword in phrase_lower)
        keyword_score = min(1.0, keyword_matches / len(topic_keywords)) if topic_keywords else 0.0
        
        return max(max_similarity, keyword_score)
    
    def _get_topic_keywords(self, topic: str) -> List[str]:
        """Get relevant keywords for a topic."""
        topic_keywords = {
            "dogs": ["dog", "pet", "animal", "walk", "feed", "play", "bark", "tail", "paw"],
            "cooking": ["cook", "food", "kitchen", "recipe", "ingredient", "bake", "fry", "boil"],
            "school": ["learn", "study", "class", "teacher", "student", "book", "homework"],
            "weather": ["sun", "rain", "cloud", "wind", "hot", "cold", "storm", "snow"],
            "birthday party": ["birthday", "party", "cake", "present", "celebrate", "candle"],
            "hospital": ["doctor", "nurse", "medicine", "sick", "heal", "patient", "treatment"],
            "shopping": ["buy", "store", "money", "cart", "checkout", "item", "price"],
            "transportation": ["travel", "vehicle", "drive", "ride", "ticket", "journey"]
        }
        
        return topic_keywords.get(topic.lower(), [])
    
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
    
    def _evaluate_diversity(self, test_item: Dict[str, Any], phrases: List[str]) -> float:
        """Evaluate the diversity of phrase categories and types."""
        if len(phrases) <= 1:
            return 0.0
        
        # Check for different phrase types (single words vs. phrases)
        single_words = [p for p in phrases if len(p.split()) == 1]
        multi_words = [p for p in phrases if len(p.split()) > 1]
        
        type_diversity = min(1.0, (len(single_words) + len(multi_words)) / len(phrases))
        
        # Check for lexical diversity
        all_words = []
        for phrase in phrases:
            all_words.extend(phrase.lower().split())
        
        unique_words = set(all_words)
        lexical_diversity = len(unique_words) / len(all_words) if all_words else 0
        
        # Check for category diversity (based on expected categories)
        expected_categories = test_item.get("expected_categories", [])
        if expected_categories:
            category_coverage = self._estimate_category_coverage(phrases, expected_categories)
        else:
            category_coverage = 0.5  # Neutral if no expected categories
        
        # Combine diversity scores
        return (type_diversity * 0.3) + (lexical_diversity * 0.4) + (category_coverage * 0.3)
    
    def _estimate_category_coverage(self, phrases: List[str], expected_categories: List[str]) -> float:
        """Estimate how well the phrases cover different expected categories."""
        # This is a simplified heuristic approach
        # In a real implementation, you might use more sophisticated categorization
        
        category_indicators = {
            "actions": ["walk", "feed", "play", "cook", "bake", "buy", "drive", "sing", "open"],
            "descriptions": ["good", "cute", "big", "small", "hot", "cold", "sunny", "rainy"],
            "objects": ["food", "toy", "book", "car", "cake", "medicine", "ticket"],
            "people": ["teacher", "doctor", "friend", "family", "student", "nurse"],
            "emotions": ["happy", "sad", "scared", "excited", "worried", "better"],
            "places": ["school", "hospital", "store", "home", "park", "kitchen"]
        }
        
        covered_categories = set()
        
        for phrase in phrases:
            phrase_lower = phrase.lower()
            for category, indicators in category_indicators.items():
                if any(indicator in phrase_lower for indicator in indicators):
                    covered_categories.add(category)
        
        # Calculate coverage of expected categories
        expected_set = set(expected_categories)
        coverage = len(covered_categories.intersection(expected_set)) / len(expected_set) if expected_set else 0.0
        
        return coverage
    
    def _evaluate_aac_suitability(self, phrases: List[str]) -> float:
        """Evaluate how suitable the phrases are for AAC use."""
        if not phrases:
            return 0.0
        
        total_suitability = 0.0
        
        for phrase in phrases:
            phrase_suitability = 1.0
            words = phrase.split()
            
            # Check phrase length (AAC phrases should be concise)
            if len(words) == 1:
                phrase_suitability += 0.1  # Single words are great for AAC
            elif len(words) <= 3:
                phrase_suitability += 0.0  # Good length
            elif len(words) <= 5:
                phrase_suitability -= 0.1  # Acceptable
            else:
                phrase_suitability -= 0.3  # Too long for typical AAC use
            
            # Check for common, functional language
            functional_words = ["i", "you", "me", "my", "need", "want", "like", "help", "please", "thank", "yes", "no"]
            if any(word.lower() in functional_words for word in words):
                phrase_suitability += 0.1
            
            # Check for complexity (simpler is better for AAC)
            if all(len(word) <= 8 for word in words):  # Not too many long words
                phrase_suitability += 0.0
            else:
                phrase_suitability -= 0.1
            
            total_suitability += max(0.0, phrase_suitability)
        
        return total_suitability / len(phrases)
    
    def _generate_feedback(self, test_item: Dict[str, Any], phrases: List[str], metrics: Dict[str, float]) -> str:
        """Generate human-readable feedback for the response."""
        feedback_parts = []
        
        # Count feedback
        phrase_count = len(phrases)
        if phrase_count == 12:
            feedback_parts.append("Perfect! Generated exactly 12 phrases.")
        elif 10 <= phrase_count <= 14:
            feedback_parts.append(f"Good! Generated {phrase_count} phrases (close to the requested 12).")
        else:
            feedback_parts.append(f"Generated {phrase_count} phrases. Aim for exactly 12.")
        
        # Format feedback
        if metrics["format_score"] >= 0.8:
            feedback_parts.append("Good CSV formatting.")
        else:
            feedback_parts.append("CSV format could be improved.")
        
        # Relevance feedback
        if metrics["relevance_score"] >= 0.8:
            feedback_parts.append("Phrases are highly relevant to the topic.")
        elif metrics["relevance_score"] >= 0.6:
            feedback_parts.append("Most phrases are relevant to the topic.")
        else:
            feedback_parts.append("Phrases could be more relevant to the topic.")
        
        # Diversity feedback
        if metrics["diversity_score"] >= 0.7:
            feedback_parts.append("Good variety in phrase types and categories.")
        else:
            feedback_parts.append("Could use more diversity in phrase types.")
        
        # AAC suitability feedback
        if metrics["aac_suitability_score"] >= 0.8:
            feedback_parts.append("Phrases are well-suited for AAC use.")
        elif metrics["aac_suitability_score"] >= 0.6:
            feedback_parts.append("Most phrases are suitable for AAC users.")
        else:
            feedback_parts.append("Phrases could be more suitable for AAC users (shorter, more functional).")
        
        return " ".join(feedback_parts)
    
    def get_test_description(self) -> str:
        """Get description of this test case."""
        return ("Phrase Board Test - Evaluates the model's ability to generate "
                "12 relevant words/phrases for AAC phrase boards on specific topics.")
    
    def get_scoring_criteria(self) -> Dict[str, str]:
        """Get scoring criteria for this test."""
        return {
            "format_score": "Adherence to CSV format and correct number of items (12)",
            "relevance_score": "How well phrases relate to the given topic",
            "diversity_score": "Variety in phrase types, categories, and vocabulary",
            "aac_suitability_score": "How suitable phrases are for AAC users (length, functionality)"
        }
