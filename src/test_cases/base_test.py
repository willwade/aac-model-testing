"""
Base Test Case Class

This module defines the abstract base class for all AAC test cases.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
import time
import logging


class BaseTest(ABC):
    """
    Abstract base class for all AAC test cases.
    
    This class defines the interface that all test cases must implement
    and provides common functionality for test execution and scoring.
    """
    
    def __init__(self, name: str = None):
        """
        Initialize the base test case.
        
        Args:
            name: Name of the test case
        """
        self.name = name or self.__class__.__name__
        self.logger = logging.getLogger(f"AAC_Testing.{self.name}")
    
    @abstractmethod
    def get_test_data(self) -> List[Dict[str, Any]]:
        """
        Get the test data for this test case.
        
        Returns:
            List of test data dictionaries, each containing:
            - input: The input to send to the model
            - expected_output: Expected output (optional)
            - prompt: The prompt template to use
            - metadata: Additional metadata for evaluation
        """
        pass
    
    @abstractmethod
    def evaluate_response(self, test_item: Dict[str, Any], response: str) -> Dict[str, Any]:
        """
        Evaluate a model's response to a test item.
        
        Args:
            test_item: The test item dictionary
            response: The model's response
        
        Returns:
            Dictionary containing evaluation results:
            - score: Numerical score (0-1)
            - passed: Boolean indicating if test passed
            - feedback: Human-readable feedback
            - metrics: Additional metrics specific to the test
        """
        pass
    
    def run_test(self, model) -> Dict[str, Any]:
        """
        Run the complete test case on a model.
        
        Args:
            model: The model instance to test
        
        Returns:
            Dictionary containing complete test results
        """
        self.logger.info(f"Starting {self.name} test")
        
        test_data = self.get_test_data()
        results = {
            "test_name": self.name,
            "total_tests": len(test_data),
            "passed_tests": 0,
            "failed_tests": 0,
            "individual_results": [],
            "overall_score": 0.0,
            "execution_time": 0.0,
            "summary_metrics": {}
        }
        
        start_time = time.time()
        total_score = 0.0
        
        for i, test_item in enumerate(test_data):
            self.logger.debug(f"Running test item {i+1}/{len(test_data)}")
            
            try:
                # Execute the test
                item_result = self._run_single_test_item(model, test_item)
                results["individual_results"].append(item_result)
                
                # Update counters
                if item_result["passed"]:
                    results["passed_tests"] += 1
                else:
                    results["failed_tests"] += 1
                
                total_score += item_result["score"]
                
            except Exception as e:
                self.logger.error(f"Error in test item {i+1}: {str(e)}")
                error_result = {
                    "test_item_id": i,
                    "input": test_item.get("input", ""),
                    "error": str(e),
                    "score": 0.0,
                    "passed": False,
                    "response_time": 0.0
                }
                results["individual_results"].append(error_result)
                results["failed_tests"] += 1
        
        # Calculate overall metrics
        results["execution_time"] = time.time() - start_time
        results["overall_score"] = total_score / len(test_data) if test_data else 0.0
        results["success_rate"] = results["passed_tests"] / len(test_data) if test_data else 0.0
        
        # Generate summary metrics
        results["summary_metrics"] = self._generate_summary_metrics(results)
        
        self.logger.info(f"Completed {self.name} test - Score: {results['overall_score']:.2f}")
        return results
    
    def _run_single_test_item(self, model, test_item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single test item.
        
        Args:
            model: The model instance
            test_item: The test item dictionary
        
        Returns:
            Dictionary containing results for this test item
        """
        item_start_time = time.time()
        
        # Prepare the prompt
        prompt = self._prepare_prompt(test_item)
        
        # Get model response
        try:
            response = model.generate_response(prompt)
        except Exception as e:
            raise Exception(f"Model generation failed: {str(e)}")
        
        response_time = time.time() - item_start_time
        
        # Evaluate the response
        evaluation = self.evaluate_response(test_item, response)
        
        # Combine results
        result = {
            "test_item_id": test_item.get("id", "unknown"),
            "input": test_item.get("input", ""),
            "prompt": prompt,
            "response": response,
            "response_time": response_time,
            **evaluation
        }
        
        return result
    
    def _prepare_prompt(self, test_item: Dict[str, Any]) -> str:
        """
        Prepare the prompt for a test item.
        
        Args:
            test_item: The test item dictionary
        
        Returns:
            The formatted prompt string
        """
        prompt_template = test_item.get("prompt", "")
        input_text = test_item.get("input", "")
        
        # Simple template substitution
        prompt = prompt_template.replace("{input}", input_text)
        
        return prompt
    
    def _generate_summary_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate summary metrics from individual test results.
        
        Args:
            results: The complete test results
        
        Returns:
            Dictionary of summary metrics
        """
        individual_results = results.get("individual_results", [])
        
        if not individual_results:
            return {}
        
        # Calculate response time statistics
        response_times = [r.get("response_time", 0) for r in individual_results if "error" not in r]
        
        summary = {
            "avg_response_time": sum(response_times) / len(response_times) if response_times else 0,
            "min_response_time": min(response_times) if response_times else 0,
            "max_response_time": max(response_times) if response_times else 0,
            "total_response_time": sum(response_times)
        }
        
        # Calculate score statistics
        scores = [r.get("score", 0) for r in individual_results if "error" not in r]
        if scores:
            summary.update({
                "avg_score": sum(scores) / len(scores),
                "min_score": min(scores),
                "max_score": max(scores),
                "score_std": self._calculate_std(scores)
            })
        
        return summary
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation of a list of values."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    def get_test_description(self) -> str:
        """
        Get a human-readable description of this test case.
        
        Returns:
            Description string
        """
        return f"{self.name} - Base test case"
    
    def get_scoring_criteria(self) -> Dict[str, str]:
        """
        Get the scoring criteria for this test case.
        
        Returns:
            Dictionary mapping criteria names to descriptions
        """
        return {
            "accuracy": "How accurate the response is",
            "relevance": "How relevant the response is to the input",
            "completeness": "How complete the response is"
        }
