#!/usr/bin/env python3
"""
View AAC Model Testing Results

This script helps you view and analyze the history of AAC model test results.
All results are stored in the 'results' directory with timestamps.

Usage:
    python view_results.py                    # Show recent results summary
    python view_results.py --history         # Show all test history
    python view_results.py --compare         # Compare latest results
    python view_results.py --device my-laptop # Filter by device
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any


def load_results_history(results_dir: Path) -> List[Dict[str, Any]]:
    """Load all results from the master results file."""
    master_file = results_dir / "all_results.jsonl"
    
    if not master_file.exists():
        return []
    
    results = []
    try:
        with open(master_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
    except Exception as e:
        print(f"Error loading results history: {e}")
        return []
    
    return results


def format_timestamp(timestamp_str: str) -> str:
    """Format timestamp for display."""
    try:
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except:
        return timestamp_str


def show_recent_summary(results_dir: Path, limit: int = 5):
    """Show summary of recent test runs."""
    history = load_results_history(results_dir)
    
    if not history:
        print("No test results found. Run some tests first!")
        return
    
    print("="*80)
    print("RECENT AAC MODEL TEST RESULTS")
    print("="*80)
    
    # Show most recent results
    recent = history[-limit:] if len(history) > limit else history
    recent.reverse()  # Most recent first
    
    for i, entry in enumerate(recent):
        print(f"\n{i+1}. {format_timestamp(entry['timestamp'])}")
        print(f"   Device: {entry.get('device_name', 'unknown')}")
        print(f"   Models: {', '.join(entry['models'])}")
        print(f"   Test Cases: {', '.join(entry['test_cases'])}")
        print(f"   Results File: {entry['results_file']}")
        
        # Show model scores
        for model, summary in entry.get('summary', {}).items():
            if summary and 'overall_score' in summary:
                score = summary['overall_score']
                time_avg = summary.get('average_response_time', 0)
                print(f"     {model}: {score:.3f} score, {time_avg:.1f}s avg time")


def show_full_history(results_dir: Path, device_filter: str = None):
    """Show complete test history."""
    history = load_results_history(results_dir)
    
    if not history:
        print("No test results found.")
        return
    
    # Filter by device if specified
    if device_filter:
        history = [h for h in history if h.get('device_name', '').lower() == device_filter.lower()]
        if not history:
            print(f"No results found for device: {device_filter}")
            return
    
    print("="*80)
    print(f"COMPLETE AAC MODEL TEST HISTORY ({len(history)} runs)")
    if device_filter:
        print(f"Filtered by device: {device_filter}")
    print("="*80)
    
    for i, entry in enumerate(history, 1):
        print(f"\n{i:3d}. {format_timestamp(entry['timestamp'])}")
        print(f"     Device: {entry.get('device_name', 'unknown')}")
        print(f"     Models: {', '.join(entry['models'])}")
        print(f"     Cases: {', '.join(entry['test_cases'])}")
        
        # Show best model for this run
        best_model = None
        best_score = 0
        for model, summary in entry.get('summary', {}).items():
            if summary and 'overall_score' in summary:
                score = summary['overall_score']
                if score > best_score:
                    best_score = score
                    best_model = model
        
        if best_model:
            print(f"     Best: {best_model} ({best_score:.3f})")


def compare_latest_results(results_dir: Path, count: int = 2):
    """Compare the latest test results."""
    history = load_results_history(results_dir)
    
    if len(history) < count:
        print(f"Need at least {count} test runs for comparison. Found {len(history)}.")
        return
    
    latest = history[-count:]
    
    print("="*80)
    print(f"COMPARISON OF LATEST {count} TEST RUNS")
    print("="*80)
    
    for i, entry in enumerate(latest):
        print(f"\nRun {i+1}: {format_timestamp(entry['timestamp'])}")
        print(f"Device: {entry.get('device_name', 'unknown')}")
        print(f"Models: {', '.join(entry['models'])}")
        
        print("\nModel Performance:")
        print(f"{'Model':<20} {'Score':<8} {'Time (s)':<10} {'Memory (MB)':<12}")
        print("-" * 50)
        
        for model, summary in entry.get('summary', {}).items():
            if summary:
                score = summary.get('overall_score', 0)
                time_avg = summary.get('average_response_time', 0)
                memory_avg = summary.get('average_memory_usage', 0)
                print(f"{model:<20} {score:<8.3f} {time_avg:<10.1f} {memory_avg:<12.1f}")


def list_available_files(results_dir: Path):
    """List all available result files."""
    if not results_dir.exists():
        print("Results directory not found.")
        return
    
    print("="*80)
    print("AVAILABLE RESULT FILES")
    print("="*80)
    
    # Group files by type
    files_by_type = {
        'Raw Results': list(results_dir.glob('raw_results_*.json')),
        'Reports': list(results_dir.glob('report_*.md')),
        'CSV Summaries': list(results_dir.glob('model_summary_*.csv')),
        'Analysis': list(results_dir.glob('analysis_*.json')),
    }
    
    for file_type, files in files_by_type.items():
        if files:
            print(f"\n{file_type}:")
            for file in sorted(files, reverse=True):  # Most recent first
                size_kb = file.stat().st_size / 1024
                mtime = datetime.fromtimestamp(file.stat().st_mtime)
                print(f"  {file.name:<40} {size_kb:>6.1f}KB  {mtime.strftime('%Y-%m-%d %H:%M')}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="View and analyze AAC model testing results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python view_results.py                    # Show recent results summary
  python view_results.py --history         # Show complete history
  python view_results.py --compare         # Compare latest 2 runs
  python view_results.py --compare 3       # Compare latest 3 runs
  python view_results.py --device work-laptop  # Filter by device
  python view_results.py --files           # List all result files
        """
    )
    
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Results directory (default: results)"
    )
    
    parser.add_argument(
        "--history",
        action="store_true",
        help="Show complete test history"
    )
    
    parser.add_argument(
        "--compare",
        type=int,
        nargs='?',
        const=2,
        help="Compare latest N test runs (default: 2)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        help="Filter results by device name"
    )
    
    parser.add_argument(
        "--files",
        action="store_true",
        help="List all available result files"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Limit number of recent results to show (default: 5)"
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    if args.files:
        list_available_files(results_dir)
    elif args.history:
        show_full_history(results_dir, args.device)
    elif args.compare:
        compare_latest_results(results_dir, args.compare)
    else:
        show_recent_summary(results_dir, args.limit)


if __name__ == "__main__":
    main()
