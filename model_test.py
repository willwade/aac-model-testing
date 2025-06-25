#!/usr/bin/env python3
"""
AAC Model Testing Script

Test locally installed language models for AAC (Augmentative and Alternative Communication) use cases.
Automatically detects locally installed Ollama models and uses them as defaults.

Usage:
    python model_test.py                           # Test all locally installed models
    python model_test.py --models gemma3:1b-it-qat tinyllama:1.1b  # Test specific models
    python model_test.py --test-cases text_correction              # Test specific cases
    python model_test.py --device-name my-laptop                   # Set device name
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from aac_testing_framework import AACTestingFramework
from model_manager import ModelManager


def get_default_models():
    """Get locally installed models suitable for AAC testing."""
    try:
        model_manager = ModelManager(verbose=False)
        local_models = model_manager.get_locally_installed_models()
        
        if not local_models:
            print("⚠️  No suitable AAC models found locally.")
            print("   Consider installing: ollama pull gemma3:1b-it-qat")
            print("                       ollama pull tinyllama:1.1b")
            return []
        
        return local_models
    except Exception as e:
        print(f"⚠️  Could not detect local models: {str(e)}")
        return []


def main():
    """Main entry point for AAC model testing."""
    parser = argparse.ArgumentParser(
        description="Test locally installed language models for AAC use cases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python model_test.py                                    # Test all local models
  python model_test.py --models gemma3:1b-it-qat         # Test specific model
  python model_test.py --test-cases text_correction      # Test specific case
  python model_test.py --device-name work-laptop         # Set device name
  python model_test.py --verbose                         # Enable verbose output
        """
    )
    
    # Get default models from local installation
    default_models = get_default_models()
    
    parser.add_argument(
        "--models", 
        nargs="+", 
        default=default_models,
        help=f"Models to test (default: locally installed models: {', '.join(default_models) if default_models else 'none found'})"
    )
    
    parser.add_argument(
        "--test-cases",
        nargs="+",
        choices=["text_correction", "utterance_suggestions", "phrase_boards", "all"],
        default=["all"],
        help="Test cases to run (default: all)"
    )
    
    parser.add_argument(
        "--device-name",
        type=str,
        default="work-laptop",
        help="Name of the device running tests (default: work-laptop)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save results (default: results)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Validate models
    if not args.models:
        print("❌ No models specified and no suitable local models found.")
        print("   Install models with: ollama pull <model-name>")
        print("   Recommended: gemma3:1b-it-qat, tinyllama:1.1b")
        return 1
    
    # Set device name in environment
    os.environ['AAC_DEVICE_NAME'] = args.device_name
    
    # Use consistent results directory
    # Results will be timestamped within the directory
    
    # Display test configuration
    print("="*80)
    print("AAC MODEL TESTING FRAMEWORK")
    print("="*80)
    print(f"Device: {args.device_name}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Models to test: {', '.join(args.models)}")
    print(f"Test cases: {', '.join(args.test_cases)}")
    print(f"Output directory: {args.output_dir}")
    print("="*80)
    
    try:
        # Initialize the framework
        print("\n1. Initializing framework...")
        framework = AACTestingFramework(
            models=args.models,
            output_dir=args.output_dir,
            verbose=args.verbose,
            performance_monitoring=True
        )
        
        print("✅ Framework initialized successfully!")
        
        # Display test case information
        print("\n2. Test Cases:")
        if "all" in args.test_cases or "text_correction" in args.test_cases:
            print("   📝 Text Correction - Fix grammatically incorrect AAC user input")
        if "all" in args.test_cases or "utterance_suggestions" in args.test_cases:
            print("   💬 Utterance Suggestions - Generate phrases from keywords")
        if "all" in args.test_cases or "phrase_boards" in args.test_cases:
            print("   📋 Phrase Boards - Create topic-based word/phrase lists")
        
        # Run tests
        print("\n3. Running tests...")
        results = framework.run_tests(test_cases=args.test_cases)
        
        print("✅ All tests completed successfully!")
        
        # Display results summary
        print("\n4. Results Summary:")
        print("-" * 60)
        
        model_summaries = {}
        for model_name, model_data in results.get("results", {}).items():
            if "error" in model_data:
                print(f"❌ {model_name}: {model_data['error']}")
                continue
                
            summary = model_data.get("summary", {})
            model_summaries[model_name] = summary
            
            print(f"\n🤖 {model_name}")
            print(f"   Overall Score:     {summary.get('overall_score', 0):.3f}")
            print(f"   Tests Passed:      {summary.get('successful_tests', 0)}/{summary.get('total_tests', 0)}")
            print(f"   Avg Response Time: {summary.get('average_response_time', 0):.2f}s")
            print(f"   Avg Memory Usage:  {summary.get('average_memory_usage', 0):.1f}MB")
            
            # AAC suitability assessment
            score = summary.get('overall_score', 0)
            time = summary.get('average_response_time', 0)
            
            if score >= 0.7 and time <= 5.0:
                print("   ✅ Recommended for AAC use")
            elif score >= 0.6 and time <= 10.0:
                print("   ⚠️  Acceptable for AAC use with limitations")
            else:
                print("   ❌ Not recommended for AAC use")
        
        # Model comparison for multiple models
        if len(model_summaries) > 1:
            print("\n5. Model Comparison:")
            print("-" * 60)
            
            # Find best overall model
            best_model = max(model_summaries.items(), 
                           key=lambda x: x[1].get('overall_score', 0))
            print(f"🏆 Best Overall: {best_model[0]} (score: {best_model[1].get('overall_score', 0):.3f})")
            
            # Find fastest model
            fastest_model = min(model_summaries.items(), 
                              key=lambda x: x[1].get('average_response_time', float('inf')))
            print(f"⚡ Fastest: {fastest_model[0]} ({fastest_model[1].get('average_response_time', 0):.2f}s)")
            
            # Find most memory efficient
            efficient_model = min(model_summaries.items(), 
                                key=lambda x: x[1].get('average_memory_usage', float('inf')))
            print(f"💾 Most Efficient: {efficient_model[0]} ({efficient_model[1].get('average_memory_usage', 0):.1f}MB)")
        
        # Generate reports
        print("\n6. Generating reports...")
        framework.generate_report(results)
        
        print("✅ Reports generated successfully!")
        print(f"\n📁 Results saved to: {args.output_dir}/")
        print("   - Detailed markdown report")
        print("   - CSV summary for analysis")
        print("   - Raw JSON results with device metadata")
        print("   - Performance monitoring logs")
        
        print("\n" + "="*80)
        print("TESTING COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Testing interrupted by user.")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
