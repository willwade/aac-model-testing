"""
AAC Testing Framework - Main Framework Class

This module contains the main AACTestingFramework class that orchestrates
the testing of language models for AAC use cases.
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from test_cases import TextCorrectionTest, UtteranceSuggestionTest, PhraseboardTest
from performance_monitor import PerformanceMonitor
from model_manager import ModelManager
from results_analyzer import ResultsAnalyzer


class AACTestingFramework:
    """
    Main framework class for testing language models on AAC use cases.
    
    This class orchestrates the entire testing process including:
    - Model management and initialization
    - Test case execution
    - Performance monitoring
    - Results collection and analysis
    - Report generation
    """
    
    def __init__(
        self,
        models: List[str],
        output_dir: str = "results",
        verbose: bool = False,
        performance_monitoring: bool = True
    ):
        """
        Initialize the AAC Testing Framework.
        
        Args:
            models: List of model names to test
            output_dir: Directory to save results
            verbose: Enable verbose logging
            performance_monitoring: Enable performance monitoring
        """
        self.models = models
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        self.performance_monitoring = performance_monitoring
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize components
        self.model_manager = ModelManager(verbose=verbose)
        self.performance_monitor = PerformanceMonitor() if performance_monitoring else None
        self.results_analyzer = ResultsAnalyzer()
        
        # Initialize test cases
        self.test_cases = {
            "text_correction": TextCorrectionTest(),
            "utterance_suggestions": UtteranceSuggestionTest(),
            "phrase_boards": PhraseboardTest()
        }
        
        self.logger.info(f"AAC Testing Framework initialized with {len(models)} models")
        self.logger.info(f"Output directory: {self.output_dir}")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_level = logging.DEBUG if self.verbose else logging.INFO
        
        # Create logs directory
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # Setup logger
        self.logger = logging.getLogger("AAC_Testing_Framework")
        self.logger.setLevel(log_level)
        
        # File handler
        log_file = log_dir / f"aac_testing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def run_tests(self, test_cases: List[str] = None) -> Dict[str, Any]:
        """
        Run the specified test cases on all models.
        
        Args:
            test_cases: List of test case names to run. If None or contains "all", 
                       runs all available test cases.
        
        Returns:
            Dictionary containing all test results
        """
        if test_cases is None or "all" in test_cases:
            test_cases_to_run = list(self.test_cases.keys())
        else:
            test_cases_to_run = test_cases
        
        self.logger.info(f"Starting tests for {len(self.models)} models")
        self.logger.info(f"Test cases to run: {test_cases_to_run}")
        
        all_results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "models": self.models,
                "test_cases": test_cases_to_run,
                "framework_version": "1.0.0",
                "device_info": self._get_device_info()
            },
            "results": {}
        }
        
        for model_name in self.models:
            self.logger.info(f"Testing model: {model_name}")
            model_results = self._test_single_model(model_name, test_cases_to_run)
            all_results["results"][model_name] = model_results
        
        # Save raw results with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = self.output_dir / f"raw_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        # Also save to a master results file for easy tracking
        master_results_file = self.output_dir / "all_results.jsonl"
        with open(master_results_file, 'a', encoding='utf-8') as f:
            # Add a single line entry for this test run
            summary_entry = {
                "timestamp": all_results["metadata"]["timestamp"],
                "models": all_results["metadata"]["models"],
                "test_cases": all_results["metadata"]["test_cases"],
                "device_name": all_results["metadata"]["device_info"].get("device_name", "unknown"),
                "results_file": results_file.name,
                "summary": {model: data.get("summary", {}) for model, data in all_results["results"].items()}
            }
            f.write(json.dumps(summary_entry, ensure_ascii=False) + '\n')

        self.logger.info(f"Raw results saved to: {results_file}")
        self.logger.info(f"Summary appended to: {master_results_file}")
        return all_results
    
    def _test_single_model(self, model_name: str, test_cases_to_run: List[str]) -> Dict[str, Any]:
        """
        Test a single model on the specified test cases.
        
        Args:
            model_name: Name of the model to test
            test_cases_to_run: List of test case names to run
        
        Returns:
            Dictionary containing results for this model
        """
        model_results = {
            "model_name": model_name,
            "test_results": {},
            "performance_metrics": {},
            "summary": {}
        }
        
        try:
            # Initialize model
            model = self.model_manager.get_model(model_name)
            
            for test_case_name in test_cases_to_run:
                self.logger.info(f"Running {test_case_name} test for {model_name}")
                
                # Start performance monitoring
                if self.performance_monitor:
                    self.performance_monitor.start_monitoring()
                
                # Run test case
                test_case = self.test_cases[test_case_name]
                test_result = test_case.run_test(model)
                
                # Stop performance monitoring
                if self.performance_monitor:
                    perf_metrics = self.performance_monitor.stop_monitoring()
                    model_results["performance_metrics"][test_case_name] = perf_metrics
                
                model_results["test_results"][test_case_name] = test_result
                
                self.logger.info(f"Completed {test_case_name} test for {model_name}")
            
            # Generate summary
            model_results["summary"] = self._generate_model_summary(model_results)
            
        except Exception as e:
            self.logger.error(f"Error testing model {model_name}: {str(e)}")
            model_results["error"] = str(e)
        
        return model_results
    
    def _generate_model_summary(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of results for a single model."""
        summary = {
            "total_tests": len(model_results["test_results"]),
            "successful_tests": 0,
            "average_response_time": 0,
            "average_memory_usage": 0,
            "overall_score": 0
        }
        
        total_response_time = 0
        total_memory_usage = 0
        total_score = 0
        
        for test_name, test_result in model_results["test_results"].items():
            if "error" not in test_result:
                summary["successful_tests"] += 1
                
                # Aggregate performance metrics
                if test_name in model_results["performance_metrics"]:
                    perf = model_results["performance_metrics"][test_name]
                    total_response_time += perf.get("response_time", 0)
                    total_memory_usage += perf.get("peak_memory_mb", 0)
                
                # Aggregate test scores
                total_score += test_result.get("overall_score", 0)
        
        if summary["successful_tests"] > 0:
            summary["average_response_time"] = total_response_time / summary["successful_tests"]
            summary["average_memory_usage"] = total_memory_usage / summary["successful_tests"]
            summary["overall_score"] = total_score / summary["successful_tests"]
        
        return summary
    
    def generate_report(self, results: Dict[str, Any]):
        """
        Generate comprehensive reports from test results.
        
        Args:
            results: Complete test results dictionary
        """
        self.logger.info("Generating comprehensive reports...")
        
        # Generate analysis
        analysis = self.results_analyzer.analyze_results(results)
        
        # Save analysis
        analysis_file = self.output_dir / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        # Generate markdown report
        self._generate_markdown_report(results, analysis)
        
        # Generate CSV summaries
        self._generate_csv_summaries(results)
        
        self.logger.info("Report generation completed")
    
    def _generate_markdown_report(self, results: Dict[str, Any], analysis: Dict[str, Any]):
        """Generate a markdown report."""
        report_file = self.output_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# AAC Model Testing Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            f.write(f"- **Models Tested:** {len(results['results'])}\n")
            f.write(f"- **Test Cases:** {', '.join(results['metadata']['test_cases'])}\n")
            f.write(f"- **Best Performing Model:** {analysis.get('best_model', 'N/A')}\n\n")
            
            # Model Comparison
            f.write("## Model Comparison\n\n")
            f.write("| Model | Overall Score | Avg Response Time (s) | Avg Memory (MB) |\n")
            f.write("|-------|---------------|----------------------|----------------|\n")
            
            for model_name, model_data in results["results"].items():
                if "summary" in model_data:
                    summary = model_data["summary"]
                    f.write(f"| {model_name} | {summary.get('overall_score', 0):.2f} | "
                           f"{summary.get('average_response_time', 0):.2f} | "
                           f"{summary.get('average_memory_usage', 0):.1f} |\n")
            
            f.write("\n")
            
            # Detailed Results
            f.write("## Detailed Results\n\n")
            for model_name, model_data in results["results"].items():
                f.write(f"### {model_name}\n\n")
                
                if "error" in model_data:
                    f.write(f"**Error:** {model_data['error']}\n\n")
                    continue
                
                for test_name, test_result in model_data.get("test_results", {}).items():
                    f.write(f"#### {test_name.replace('_', ' ').title()}\n\n")
                    
                    if "error" in test_result:
                        f.write(f"**Error:** {test_result['error']}\n\n")
                    else:
                        f.write(f"- **Score:** {test_result.get('overall_score', 0):.2f}\n")
                        f.write(f"- **Test Cases Passed:** {test_result.get('passed_tests', 0)}/{test_result.get('total_tests', 0)}\n")
                        
                        # Performance metrics
                        if test_name in model_data.get("performance_metrics", {}):
                            perf = model_data["performance_metrics"][test_name]
                            f.write(f"- **Response Time:** {perf.get('response_time', 0):.2f}s\n")
                            f.write(f"- **Memory Usage:** {perf.get('peak_memory_mb', 0):.1f}MB\n")
                        
                        f.write("\n")
        
        self.logger.info(f"Markdown report saved to: {report_file}")
    
    def _generate_csv_summaries(self, results: Dict[str, Any]):
        """Generate CSV summaries for easy analysis."""
        import csv
        
        # Model summary CSV
        summary_file = self.output_dir / f"model_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        with open(summary_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Model", "Overall Score", "Successful Tests", "Total Tests",
                "Avg Response Time (s)", "Avg Memory Usage (MB)"
            ])
            
            for model_name, model_data in results["results"].items():
                if "summary" in model_data and "error" not in model_data:
                    summary = model_data["summary"]
                    writer.writerow([
                        model_name,
                        f"{summary.get('overall_score', 0):.2f}",
                        summary.get('successful_tests', 0),
                        summary.get('total_tests', 0),
                        f"{summary.get('average_response_time', 0):.2f}",
                        f"{summary.get('average_memory_usage', 0):.1f}"
                    ])
        
        self.logger.info(f"CSV summary saved to: {summary_file}")

    def _get_device_info(self) -> Dict[str, Any]:
        """Get information about the device running the tests."""
        import platform
        import psutil
        import socket

        try:
            device_info = {
                "device_name": "work-laptop",  # Default device name
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "system": platform.system(),
                "processor": platform.processor(),
                "architecture": platform.architecture()[0],
                "python_version": platform.python_version(),
                "cpu_count": psutil.cpu_count(),
                "cpu_count_logical": psutil.cpu_count(logical=True),
                "total_memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                "available_memory_gb": round(psutil.virtual_memory().available / (1024**3), 2)
            }

            # Try to get more specific device name from environment or hostname
            import os
            env_device_name = os.environ.get('AAC_DEVICE_NAME')
            if env_device_name:
                device_info["device_name"] = env_device_name
            elif "laptop" in socket.gethostname().lower():
                device_info["device_name"] = "work-laptop"
            else:
                device_info["device_name"] = socket.gethostname()

            return device_info

        except Exception as e:
            self.logger.warning(f"Failed to get complete device info: {str(e)}")
            return {
                "device_name": "work-laptop",
                "error": str(e)
            }
