"""
Results Analyzer

This module provides comprehensive analysis of test results, including
statistical analysis, model comparison, and performance insights.
"""

import logging
from typing import Dict, Any, List, Tuple
from datetime import datetime
import statistics


class ResultsAnalyzer:
    """
    Comprehensive analyzer for AAC model testing results.
    
    This class provides statistical analysis, model comparison,
    and insights generation from test results.
    """
    
    def __init__(self):
        """Initialize the results analyzer."""
        self.logger = logging.getLogger("AAC_Testing.ResultsAnalyzer")
    
    def analyze_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of test results.
        
        Args:
            results: Complete test results dictionary
        
        Returns:
            Dictionary containing comprehensive analysis
        """
        self.logger.info("Starting comprehensive results analysis")
        
        analysis = {
            "analysis_timestamp": datetime.now().isoformat(),
            "summary": self._generate_summary(results),
            "model_rankings": self._rank_models(results),
            "test_case_analysis": self._analyze_test_cases(results),
            "performance_analysis": self._analyze_performance(results),
            "recommendations": self._generate_recommendations(results),
            "statistical_insights": self._generate_statistical_insights(results)
        }
        
        self.logger.info("Results analysis completed")
        return analysis
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate high-level summary of results."""
        model_results = results.get("results", {})
        
        total_models = len(model_results)
        successful_models = sum(1 for model_data in model_results.values() 
                               if "error" not in model_data)
        
        # Calculate overall statistics
        all_scores = []
        all_response_times = []
        all_memory_usage = []
        
        for model_data in model_results.values():
            if "summary" in model_data and "error" not in model_data:
                summary = model_data["summary"]
                all_scores.append(summary.get("overall_score", 0))
                all_response_times.append(summary.get("average_response_time", 0))
                all_memory_usage.append(summary.get("average_memory_usage", 0))
        
        return {
            "total_models_tested": total_models,
            "successful_models": successful_models,
            "failed_models": total_models - successful_models,
            "test_cases_run": len(results.get("metadata", {}).get("test_cases", [])),
            "overall_statistics": {
                "average_score": statistics.mean(all_scores) if all_scores else 0,
                "score_std_dev": statistics.stdev(all_scores) if len(all_scores) > 1 else 0,
                "average_response_time": statistics.mean(all_response_times) if all_response_times else 0,
                "average_memory_usage": statistics.mean(all_memory_usage) if all_memory_usage else 0
            }
        }
    
    def _rank_models(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Rank models by overall performance."""
        model_results = results.get("results", {})
        
        model_scores = []
        
        for model_name, model_data in model_results.items():
            if "error" in model_data:
                continue
            
            summary = model_data.get("summary", {})
            
            # Calculate composite score
            overall_score = summary.get("overall_score", 0)
            response_time = summary.get("average_response_time", float('inf'))
            memory_usage = summary.get("average_memory_usage", float('inf'))
            success_rate = summary.get("successful_tests", 0) / max(summary.get("total_tests", 1), 1)
            
            # Normalize metrics (lower is better for time and memory)
            # Use harmonic mean for response time penalty (avoid division by zero)
            time_penalty = 1 / (1 + response_time) if response_time > 0 else 1
            memory_penalty = 1 / (1 + memory_usage / 1000) if memory_usage > 0 else 1  # Scale memory
            
            # Composite score (weighted)
            composite_score = (
                overall_score * 0.5 +  # Test performance (50%)
                success_rate * 0.2 +   # Reliability (20%)
                time_penalty * 0.15 +  # Speed (15%)
                memory_penalty * 0.15  # Efficiency (15%)
            )
            
            model_scores.append({
                "model_name": model_name,
                "composite_score": composite_score,
                "overall_score": overall_score,
                "success_rate": success_rate,
                "average_response_time": response_time,
                "average_memory_usage": memory_usage,
                "rank": 0  # Will be set after sorting
            })
        
        # Sort by composite score (descending)
        model_scores.sort(key=lambda x: x["composite_score"], reverse=True)
        
        # Assign ranks
        for i, model_score in enumerate(model_scores):
            model_score["rank"] = i + 1
        
        return model_scores
    
    def _analyze_test_cases(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance across different test cases."""
        model_results = results.get("results", {})
        test_case_analysis = {}
        
        # Collect data for each test case
        for model_name, model_data in model_results.items():
            if "error" in model_data:
                continue
            
            test_results = model_data.get("test_results", {})
            
            for test_case_name, test_result in test_results.items():
                if "error" in test_result:
                    continue
                
                if test_case_name not in test_case_analysis:
                    test_case_analysis[test_case_name] = {
                        "scores": [],
                        "response_times": [],
                        "success_rates": [],
                        "model_performances": {}
                    }
                
                # Collect metrics
                score = test_result.get("overall_score", 0)
                success_rate = test_result.get("passed_tests", 0) / max(test_result.get("total_tests", 1), 1)
                
                test_case_analysis[test_case_name]["scores"].append(score)
                test_case_analysis[test_case_name]["success_rates"].append(success_rate)
                test_case_analysis[test_case_name]["model_performances"][model_name] = {
                    "score": score,
                    "success_rate": success_rate
                }
                
                # Add response time if available
                if model_name in model_data.get("performance_metrics", {}):
                    perf_metrics = model_data["performance_metrics"][test_case_name]
                    response_time = perf_metrics.get("response_time", 0)
                    test_case_analysis[test_case_name]["response_times"].append(response_time)
        
        # Calculate statistics for each test case
        for test_case_name, data in test_case_analysis.items():
            scores = data["scores"]
            response_times = data["response_times"]
            success_rates = data["success_rates"]
            
            data["statistics"] = {
                "average_score": statistics.mean(scores) if scores else 0,
                "score_std_dev": statistics.stdev(scores) if len(scores) > 1 else 0,
                "min_score": min(scores) if scores else 0,
                "max_score": max(scores) if scores else 0,
                "average_success_rate": statistics.mean(success_rates) if success_rates else 0,
                "average_response_time": statistics.mean(response_times) if response_times else 0,
                "models_tested": len(data["model_performances"])
            }
            
            # Find best and worst performing models for this test case
            if data["model_performances"]:
                sorted_models = sorted(
                    data["model_performances"].items(),
                    key=lambda x: x[1]["score"],
                    reverse=True
                )
                data["best_model"] = sorted_models[0][0] if sorted_models else None
                data["worst_model"] = sorted_models[-1][0] if sorted_models else None
        
        return test_case_analysis
    
    def _analyze_performance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance characteristics across models."""
        model_results = results.get("results", {})
        
        performance_data = {
            "response_times": [],
            "memory_usage": [],
            "cpu_usage": [],
            "model_efficiency": {}
        }
        
        for model_name, model_data in model_results.items():
            if "error" in model_data:
                continue
            
            summary = model_data.get("summary", {})
            perf_metrics = model_data.get("performance_metrics", {})
            
            # Collect overall performance data
            avg_response_time = summary.get("average_response_time", 0)
            avg_memory = summary.get("average_memory_usage", 0)
            
            if avg_response_time > 0:
                performance_data["response_times"].append(avg_response_time)
            if avg_memory > 0:
                performance_data["memory_usage"].append(avg_memory)
            
            # Calculate efficiency score (quality per resource unit)
            overall_score = summary.get("overall_score", 0)
            if avg_response_time > 0 and avg_memory > 0:
                efficiency_score = overall_score / (avg_response_time * (avg_memory / 1000))
                performance_data["model_efficiency"][model_name] = {
                    "efficiency_score": efficiency_score,
                    "quality_score": overall_score,
                    "response_time": avg_response_time,
                    "memory_usage": avg_memory
                }
        
        # Calculate performance statistics
        response_times = performance_data["response_times"]
        memory_usage = performance_data["memory_usage"]
        
        performance_stats = {}
        
        if response_times:
            performance_stats["response_time_stats"] = {
                "average": statistics.mean(response_times),
                "median": statistics.median(response_times),
                "std_dev": statistics.stdev(response_times) if len(response_times) > 1 else 0,
                "min": min(response_times),
                "max": max(response_times)
            }
        
        if memory_usage:
            performance_stats["memory_usage_stats"] = {
                "average": statistics.mean(memory_usage),
                "median": statistics.median(memory_usage),
                "std_dev": statistics.stdev(memory_usage) if len(memory_usage) > 1 else 0,
                "min": min(memory_usage),
                "max": max(memory_usage)
            }
        
        # Find most efficient model
        if performance_data["model_efficiency"]:
            most_efficient = max(
                performance_data["model_efficiency"].items(),
                key=lambda x: x[1]["efficiency_score"]
            )
            performance_stats["most_efficient_model"] = {
                "name": most_efficient[0],
                **most_efficient[1]
            }
        
        return {
            "statistics": performance_stats,
            "efficiency_rankings": sorted(
                performance_data["model_efficiency"].items(),
                key=lambda x: x[1]["efficiency_score"],
                reverse=True
            )
        }
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        model_rankings = self._rank_models(results)
        test_case_analysis = self._analyze_test_cases(results)
        
        # Best overall model recommendation
        if model_rankings:
            best_model = model_rankings[0]
            recommendations.append({
                "type": "best_overall",
                "title": "Best Overall Model",
                "description": f"{best_model['model_name']} shows the best overall performance",
                "details": {
                    "model": best_model['model_name'],
                    "composite_score": best_model['composite_score'],
                    "overall_score": best_model['overall_score'],
                    "response_time": best_model['average_response_time']
                },
                "priority": "high"
            })
        
        # Performance-specific recommendations
        if len(model_rankings) > 1:
            fastest_model = min(model_rankings, key=lambda x: x['average_response_time'])
            most_efficient_memory = min(model_rankings, key=lambda x: x['average_memory_usage'])
            
            if fastest_model['average_response_time'] > 0:
                recommendations.append({
                    "type": "fastest_response",
                    "title": "Fastest Response Time",
                    "description": f"{fastest_model['model_name']} has the fastest average response time",
                    "details": {
                        "model": fastest_model['model_name'],
                        "response_time": fastest_model['average_response_time']
                    },
                    "priority": "medium"
                })
            
            if most_efficient_memory['average_memory_usage'] > 0:
                recommendations.append({
                    "type": "memory_efficient",
                    "title": "Most Memory Efficient",
                    "description": f"{most_efficient_memory['model_name']} uses the least memory",
                    "details": {
                        "model": most_efficient_memory['model_name'],
                        "memory_usage": most_efficient_memory['average_memory_usage']
                    },
                    "priority": "medium"
                })
        
        # Test case specific recommendations
        for test_case_name, analysis in test_case_analysis.items():
            if analysis.get("best_model"):
                recommendations.append({
                    "type": "test_case_specific",
                    "title": f"Best for {test_case_name.replace('_', ' ').title()}",
                    "description": f"{analysis['best_model']} performs best on {test_case_name}",
                    "details": {
                        "model": analysis['best_model'],
                        "test_case": test_case_name,
                        "score": analysis['model_performances'][analysis['best_model']]['score']
                    },
                    "priority": "low"
                })
        
        return recommendations
    
    def _generate_statistical_insights(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate statistical insights from the results."""
        model_results = results.get("results", {})
        
        insights = {
            "correlations": {},
            "distributions": {},
            "outliers": {},
            "trends": {}
        }
        
        # Collect data for correlation analysis
        scores = []
        response_times = []
        memory_usage = []
        
        for model_data in model_results.values():
            if "error" in model_data or "summary" not in model_data:
                continue
            
            summary = model_data["summary"]
            scores.append(summary.get("overall_score", 0))
            response_times.append(summary.get("average_response_time", 0))
            memory_usage.append(summary.get("average_memory_usage", 0))
        
        # Calculate correlations if we have enough data
        if len(scores) > 2:
            try:
                # Score vs Response Time correlation
                if response_times and any(rt > 0 for rt in response_times):
                    score_time_corr = self._calculate_correlation(scores, response_times)
                    insights["correlations"]["score_vs_response_time"] = score_time_corr
                
                # Score vs Memory correlation
                if memory_usage and any(mu > 0 for mu in memory_usage):
                    score_memory_corr = self._calculate_correlation(scores, memory_usage)
                    insights["correlations"]["score_vs_memory_usage"] = score_memory_corr
                
            except Exception as e:
                self.logger.warning(f"Failed to calculate correlations: {str(e)}")
        
        # Distribution analysis
        if scores:
            insights["distributions"]["score_distribution"] = {
                "mean": statistics.mean(scores),
                "median": statistics.median(scores),
                "std_dev": statistics.stdev(scores) if len(scores) > 1 else 0,
                "range": max(scores) - min(scores) if scores else 0
            }
        
        return insights
    
    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient."""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi * xi for xi in x)
        sum_y2 = sum(yi * yi for yi in y)
        
        numerator = n * sum_xy - sum_x * sum_y
        denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)) ** 0.5
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
