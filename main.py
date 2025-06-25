#!/usr/bin/env python3
"""
AAC Model Testing Framework

A comprehensive testing framework to evaluate small offline language models
for Augmentative and Alternative Communication (AAC) use cases on low-powered machines.

This framework tests models on three primary AAC use cases:
1. Text Correction for AAC Users
2. Utterance Suggestion Generation
3. Topic-Based Phrase Board Generation

Author: AAC Model Testing Framework
"""

import sys
import argparse
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from aac_testing_framework import AACTestingFramework
from model_manager import ModelManager


def get_default_models():
    """Get locally installed models suitable for AAC testing."""
    try:
        model_manager = ModelManager(verbose=False)
        local_models = model_manager.get_locally_installed_models()
        return local_models if local_models else ["gemma3:1b-it-qat"]
    except Exception:
        return ["gemma3:1b-it-qat"]


def main():
    """Main entry point for the AAC Model Testing Framework."""
    parser = argparse.ArgumentParser(
        description="AAC Model Testing Framework - Evaluate small offline language models for AAC use cases"
    )

    default_models = get_default_models()

    parser.add_argument(
        "--models",
        nargs="+",
        default=default_models,
        help=f"List of models to test (default: {' '.join(default_models)})"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save test results (default: results)"
    )

    parser.add_argument(
        "--test-cases",
        nargs="+",
        choices=["text_correction", "utterance_suggestions", "phrase_boards", "all"],
        default=["all"],
        help="Test cases to run (default: all)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    parser.add_argument(
        "--performance-monitoring",
        action="store_true",
        default=True,
        help="Enable performance monitoring (default: True)"
    )

    args = parser.parse_args()

    # Initialize the testing framework
    framework = AACTestingFramework(
        models=args.models,
        output_dir=args.output_dir,
        verbose=args.verbose,
        performance_monitoring=args.performance_monitoring
    )

    # Run the tests
    try:
        results = framework.run_tests(test_cases=args.test_cases)

        print("\n" + "="*80)
        print("AAC MODEL TESTING COMPLETED")
        print("="*80)

        framework.generate_report(results)

        print(f"\nDetailed results saved to: {args.output_dir}")

    except KeyboardInterrupt:
        print("\n\nTesting interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during testing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
