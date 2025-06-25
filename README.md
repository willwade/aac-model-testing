# AAC Model Testing Framework

A comprehensive testing framework to evaluate small offline language models for Augmentative and Alternative Communication (AAC) use cases on low-powered machines.

## Overview

This framework evaluates language models on three critical AAC use cases:

1. **Text Correction for AAC Users** - Correcting grammatically incorrect or incomplete sentences
2. **Utterance Suggestion Generation** - Generating multiple phrase suggestions from minimal input
3. **Topic-Based Phrase Board Generation** - Creating 12 relevant words/phrases for AAC phrase boards


## Requirements

- Python 3.8+
- [uv](https://github.com/astral-sh/uv) for package management
- [Ollama](https://ollama.ai/) for model backend
- Windows/Linux/macOS support

## Installation

### Option 1: Automatic Setup (Recommended)

**Windows:**
```bash
# Double-click init.bat, or run in PowerShell:
.\init.ps1

# With options:
.\init.ps1 -DeviceName "my-laptop" -Verbose
```

**Linux/macOS:**
```bash
# Make executable and run:
chmod +x init.sh
./init.sh

# With options:
./init.sh --device-name my-laptop --verbose
```

### Option 2: Manual Setup

**1. Install Prerequisites:**
```bash
# Install uv: https://github.com/astral-sh/uv
# Install Ollama: https://ollama.ai/
```

**2. Clone and Setup:**
```bash
git clone <repository-url>
cd aac-model-testing
uv sync
uv run llm install llm-ollama
```

**3. Download Models:**
```bash
ollama pull gemma3:1b-it-qat
ollama pull tinyllama:1.1b
```

## Quick Start

### Basic Usage

```bash
# Test all locally installed models (recommended)
uv run python model_test.py

# Test specific models
uv run python model_test.py --models gemma3:1b-it-qat tinyllama:1.1b

# Test specific cases only
uv run python model_test.py --test-cases text_correction

# Set device name for tracking
uv run python model_test.py --device-name my-laptop

# Enable verbose output
uv run python model_test.py --verbose
```

### Command Line Options

```bash
uv run python model_test.py --help
```

**Available Options:**
- `--models`: Models to test (default: auto-detected locally installed models)
- `--test-cases`: Test cases to run (text_correction, utterance_suggestions, phrase_boards, all)
- `--device-name`: Name of device for tracking (default: work-laptop)
- `--output-dir`: Directory for results (default: auto-generated with timestamp)
- `--verbose`: Enable verbose output

**Note:** The framework automatically detects locally installed Ollama models and only tests those. No online models are used.

## Test Cases

### 1. Text Correction Test

**Purpose:** Evaluate ability to correct poorly written AAC user sentences.

**Example:**
- **Input:** "me want eat pizza now hungry"
- **Expected:** "I want to eat pizza now because I'm hungry."

**Evaluation Criteria:**
- Grammar correctness (30%)
- Completeness (25%)
- Naturalness (25%)
- Semantic preservation (20%)

### 2. Utterance Suggestion Test

**Purpose:** Generate multiple phrase suggestions from minimal input.

**Example:**
- **Input:** "pizza vegetarian"
- **Expected Output:**
  - "I want a vegetarian pizza"
  - "Do you have vegetarian pizza?"
  - "I love vegetarian pizza"

**Evaluation Criteria:**
- Completeness (appropriate number of phrases)
- Relevance to input keywords
- Diversity in phrase structures
- Overall quality and grammar

### 3. Phrase Board Test

**Purpose:** Generate 12 relevant words/phrases for AAC phrase boards.

**Example:**
- **Input:** "dogs"
- **Expected CSV Output:**
  ```
  walk the dog
  feed the dog
  good dog
  dog food
  ...
  ```

**Evaluation Criteria:**
- Format adherence (CSV with 12 items)
- Topic relevance
- Category diversity
- AAC suitability (concise, functional)

## Initialization Scripts

The framework includes automatic setup scripts for easy installation:

### Windows
- **`init.ps1`** - PowerShell script with full functionality
- **`init.bat`** - Simple batch file that calls the PowerShell script

### Linux/macOS
- **`init.sh`** - Bash script for Unix-like systems

### Script Options
```bash
# Windows PowerShell
.\init.ps1 -DeviceName "my-laptop" -SkipTests -Verbose

# Linux/macOS Bash
./init.sh --device-name my-laptop --skip-tests --verbose
```

## Understanding Results

### Output Files

All results are saved in a single `results/` directory with timestamps:
```
results/
├── all_results.jsonl                    # Master log of all test runs
├── logs/
│   └── aac_testing_YYYYMMDD_HHMMSS.log
├── raw_results_YYYYMMDD_HHMMSS.json     # Detailed results
├── analysis_YYYYMMDD_HHMMSS.json        # Statistical analysis
├── report_YYYYMMDD_HHMMSS.md            # Human-readable report
└── model_summary_YYYYMMDD_HHMMSS.csv    # CSV for spreadsheet analysis
```

### Viewing Results History

```bash
# Show recent test summaries
uv run python view_results.py

# Show complete test history
uv run python view_results.py --history

# Compare latest test runs
uv run python view_results.py --compare

# Filter by device
uv run python view_results.py --device work-laptop

# List all result files
uv run python view_results.py --files
```

### Excel Summary Reports

Generate comprehensive Excel spreadsheets with all test data:

```bash
# Generate default Excel summary
uv run python generate_summary_xlsx.py

# Custom filename
uv run python generate_summary_xlsx.py --output my_report.xlsx

# Filter by device
uv run python generate_summary_xlsx.py --device work-laptop
```

### AAC Suitability Guidelines

**Recommended for AAC:**
- Overall Score ≥ 0.7
- Response Time ≤ 5 seconds
- Memory Usage suitable for target device

## Troubleshooting

### Common Issues

**"Model not found" Error:**
```bash
# Check available models
uv run llm models list
ollama list
```

**Installation Script Issues:**
```bash
# Windows: Run as Administrator if needed
# Linux/macOS: Ensure curl is installed
# Check internet connection for downloads
```

**Memory Issues:**
- Close other applications
- Test one model at a time
- Use smaller models (1B parameters or less)

**Slow Performance:**
- Check system resources
- Ensure Ollama is using appropriate hardware acceleration

### Debug Mode

```bash
# Enable maximum verbosity
uv run python model_test.py --verbose

# Check logs in results directory
```

## Contributing

To add new test cases or models:
1. Fork the repository
2. Add test cases in `src/test_cases/`
3. Update model recommendations in `src/model_manager.py`
4. Test thoroughly and submit a pull request

## License

Will Wade / MIT License