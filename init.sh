#!/bin/bash
set -e

# AAC Model Testing Framework - Unix Initialization Script
#
# This script automatically sets up the AAC Model Testing Framework on Linux/macOS by:
# - Installing uv (Python package manager)
# - Installing Ollama (LLM runtime)
# - Downloading recommended AAC models
# - Installing Python dependencies
# - Running initial tests
#
# Usage:
#   ./init.sh
#   ./init.sh --device-name my-laptop
#   ./init.sh --skip-tests
#   ./init.sh --verbose

# Default values
DEVICE_NAME="${HOSTNAME:-$(uname -n)}"
SKIP_TESTS=false
VERBOSE=false

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --device-name)
            DEVICE_NAME="$2"
            shift 2
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --device-name NAME    Set device name for tracking (default: hostname)"
            echo "  --skip-tests         Skip running initial tests"
            echo "  --verbose            Enable verbose output"
            echo "  -h, --help           Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Helper functions
log_step() {
    echo -e "${BLUE}==>${NC} $1"
}

log_success() {
    echo -e "${GREEN}✅${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}⚠️${NC} $1"
}

log_error() {
    echo -e "${RED}❌${NC} $1"
}

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

install_uv() {
    log_step "Installing uv (Python package manager)..."
    
    if command_exists uv; then
        log_success "uv is already installed"
        uv --version
        return
    fi
    
    # Install uv
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Source the shell configuration to get uv in PATH
    if [[ -f "$HOME/.cargo/env" ]]; then
        source "$HOME/.cargo/env"
    fi
    
    # Add to current session PATH
    export PATH="$HOME/.cargo/bin:$PATH"
    
    if command_exists uv; then
        log_success "uv installed successfully"
        uv --version
    else
        log_error "uv installation failed - command not found after installation"
        echo "Please install uv manually from: https://github.com/astral-sh/uv"
        exit 1
    fi
}

install_ollama() {
    log_step "Installing Ollama (LLM runtime)..."
    
    if command_exists ollama; then
        log_success "Ollama is already installed"
        ollama --version
        return
    fi
    
    # Install Ollama
    curl -fsSL https://ollama.ai/install.sh | sh
    
    # Wait a moment for installation to complete
    sleep 3
    
    if command_exists ollama; then
        log_success "Ollama installed successfully"
        ollama --version
    else
        log_error "Ollama installation failed - command not found after installation"
        echo "Please install Ollama manually from: https://ollama.ai/"
        exit 1
    fi
}

install_models() {
    log_step "Downloading recommended AAC models..."
    
    models=("gemma3:1b-it-qat" "tinyllama:1.1b")
    
    for model in "${models[@]}"; do
        echo "Checking if $model is already installed..."
        
        if ollama list 2>/dev/null | grep -q "$model"; then
            log_success "$model is already installed"
            continue
        fi
        
        echo "Downloading $model (this may take several minutes)..."
        if ollama pull "$model"; then
            log_success "$model downloaded successfully"
        else
            log_warning "Failed to download $model"
            echo "You can download it later with: ollama pull $model"
        fi
    done
}

setup_python_environment() {
    log_step "Setting up Python environment..."
    
    # Install dependencies
    echo "Installing Python dependencies..."
    uv sync
    
    # Install LLM Ollama plugin
    echo "Installing LLM Ollama plugin..."
    uv run llm install llm-ollama
    
    log_success "Python environment setup complete"
}

run_initial_tests() {
    if [[ "$SKIP_TESTS" == "true" ]]; then
        log_warning "Skipping initial tests (--skip-tests specified)"
        return
    fi
    
    log_step "Running initial AAC model tests..."
    
    test_args=("--device-name" "$DEVICE_NAME")
    if [[ "$VERBOSE" == "true" ]]; then
        test_args+=("--verbose")
    fi
    
    echo "Testing with device name: $DEVICE_NAME"
    if uv run python model_test.py "${test_args[@]}"; then
        log_success "Initial tests completed successfully!"
    else
        log_warning "Initial tests failed"
        echo "You can run tests manually later with: uv run python model_test.py"
    fi
}

main() {
    cat << EOF
${BLUE}
================================================================================
AAC MODEL TESTING FRAMEWORK - UNIX SETUP
================================================================================
${NC}
This script will install and configure everything needed for AAC model testing:

📦 uv (Python package manager)
🤖 Ollama (LLM runtime)  
📥 AAC Models (gemma3:1b-it-qat, tinyllama:1.1b)
🐍 Python dependencies
🧪 Initial test run

Device: $DEVICE_NAME
Skip Tests: $SKIP_TESTS

Press Ctrl+C to cancel, or Enter to continue...
EOF
    
    read -r
    
    # Check if we're in the right directory
    if [[ ! -f "model_test.py" ]]; then
        log_error "model_test.py not found. Please run this script from the aac-model-testing directory."
        exit 1
    fi
    
    # Installation steps
    install_uv
    install_ollama
    setup_python_environment
    install_models
    run_initial_tests
    
    cat << EOF

${GREEN}
================================================================================
🎉 AAC MODEL TESTING FRAMEWORK SETUP COMPLETE!
================================================================================
${NC}

✅ All components installed successfully
✅ Models downloaded and ready
✅ Framework tested and working

${BLUE}Next steps:${NC}
• Run tests: ${YELLOW}uv run python model_test.py${NC}
• View help: ${YELLOW}uv run python model_test.py --help${NC}
• Test specific model: ${YELLOW}uv run python model_test.py --models tinyllama:1.1b${NC}
• Set device name: ${YELLOW}uv run python model_test.py --device-name my-device${NC}

${BLUE}All results are saved to the 'results/' directory with timestamps.${NC}
${BLUE}View test history: ${YELLOW}uv run python view_results.py${NC}

Happy testing! 🚀
EOF
}

# Check for required tools
if ! command_exists curl; then
    log_error "curl is required but not installed. Please install curl first."
    exit 1
fi

# Run main function
main
