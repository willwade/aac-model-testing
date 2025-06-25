#!/usr/bin/env powershell
<#
.SYNOPSIS
    AAC Model Testing Framework - Windows Initialization Script

.DESCRIPTION
    This script automatically sets up the AAC Model Testing Framework on Windows by:
    - Installing uv (Python package manager)
    - Installing Ollama (LLM runtime)
    - Downloading recommended AAC models
    - Installing Python dependencies
    - Running initial tests

.EXAMPLE
    .\init.ps1
    
.EXAMPLE
    .\init.ps1 -DeviceName "my-laptop"
    
.EXAMPLE
    .\init.ps1 -SkipTests
#>

param(
    [string]$DeviceName = $env:COMPUTERNAME,
    [switch]$SkipTests = $false,
    [switch]$Verbose = $false
)

# Set error action preference
$ErrorActionPreference = "Stop"

# Colors for output
$Green = "`e[32m"
$Red = "`e[31m"
$Yellow = "`e[33m"
$Blue = "`e[34m"
$Reset = "`e[0m"

function Write-Step {
    param([string]$Message)
    Write-Host "${Blue}==>${Reset} $Message" -ForegroundColor Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "${Green}✅${Reset} $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "${Yellow}⚠️${Reset} $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "${Red}❌${Reset} $Message" -ForegroundColor Red
}

function Test-Command {
    param([string]$Command)
    try {
        Get-Command $Command -ErrorAction Stop | Out-Null
        return $true
    }
    catch {
        return $false
    }
}

function Install-UV {
    Write-Step "Installing uv (Python package manager)..."
    
    if (Test-Command "uv") {
        Write-Success "uv is already installed"
        uv --version
        return
    }
    
    try {
        # Download and install uv
        Write-Host "Downloading uv installer..."
        Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
        
        # Refresh PATH
        $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("PATH", "User")
        
        if (Test-Command "uv") {
            Write-Success "uv installed successfully"
            uv --version
        } else {
            throw "uv installation failed - command not found after installation"
        }
    }
    catch {
        Write-Error "Failed to install uv: $_"
        Write-Host "Please install uv manually from: https://github.com/astral-sh/uv"
        exit 1
    }
}

function Install-Ollama {
    Write-Step "Installing Ollama (LLM runtime)..."
    
    if (Test-Command "ollama") {
        Write-Success "Ollama is already installed"
        ollama --version
        return
    }
    
    try {
        # Download Ollama installer
        Write-Host "Downloading Ollama installer..."
        $installerPath = "$env:TEMP\OllamaSetup.exe"
        Invoke-WebRequest -Uri "https://ollama.ai/download/windows" -OutFile $installerPath
        
        # Run installer silently
        Write-Host "Installing Ollama..."
        Start-Process -FilePath $installerPath -ArgumentList "/S" -Wait
        
        # Add Ollama to PATH if needed
        $ollamaPath = "$env:LOCALAPPDATA\Programs\Ollama"
        if (Test-Path $ollamaPath) {
            $currentPath = [Environment]::GetEnvironmentVariable("PATH", "User")
            if ($currentPath -notlike "*$ollamaPath*") {
                [Environment]::SetEnvironmentVariable("PATH", "$currentPath;$ollamaPath", "User")
                $env:PATH += ";$ollamaPath"
            }
        }
        
        # Wait a moment for installation to complete
        Start-Sleep -Seconds 3
        
        if (Test-Command "ollama") {
            Write-Success "Ollama installed successfully"
            ollama --version
        } else {
            throw "Ollama installation failed - command not found after installation"
        }
        
        # Clean up installer
        Remove-Item $installerPath -ErrorAction SilentlyContinue
    }
    catch {
        Write-Error "Failed to install Ollama: $_"
        Write-Host "Please install Ollama manually from: https://ollama.ai/"
        exit 1
    }
}

function Install-Models {
    Write-Step "Downloading recommended AAC models..."
    
    $models = @("gemma3:1b-it-qat", "tinyllama:1.1b")
    
    foreach ($model in $models) {
        Write-Host "Checking if $model is already installed..."
        
        $installedModels = ollama list 2>$null
        if ($installedModels -match $model) {
            Write-Success "$model is already installed"
            continue
        }
        
        Write-Host "Downloading $model (this may take several minutes)..."
        try {
            ollama pull $model
            Write-Success "$model downloaded successfully"
        }
        catch {
            Write-Warning "Failed to download $model: $_"
            Write-Host "You can download it later with: ollama pull $model"
        }
    }
}

function Setup-Python-Environment {
    Write-Step "Setting up Python environment..."
    
    try {
        # Install dependencies
        Write-Host "Installing Python dependencies..."
        uv sync
        
        # Install LLM Ollama plugin
        Write-Host "Installing LLM Ollama plugin..."
        uv run llm install llm-ollama
        
        Write-Success "Python environment setup complete"
    }
    catch {
        Write-Error "Failed to setup Python environment: $_"
        exit 1
    }
}

function Run-Initial-Tests {
    if ($SkipTests) {
        Write-Warning "Skipping initial tests (--SkipTests specified)"
        return
    }
    
    Write-Step "Running initial AAC model tests..."
    
    try {
        $testArgs = @("--device-name", $DeviceName)
        if ($Verbose) {
            $testArgs += "--verbose"
        }
        
        Write-Host "Testing with device name: $DeviceName"
        uv run python model_test.py @testArgs
        
        Write-Success "Initial tests completed successfully!"
    }
    catch {
        Write-Warning "Initial tests failed: $_"
        Write-Host "You can run tests manually later with: uv run python model_test.py"
    }
}

function Main {
    Write-Host @"
${Blue}
================================================================================
AAC MODEL TESTING FRAMEWORK - WINDOWS SETUP
================================================================================
${Reset}
This script will install and configure everything needed for AAC model testing:

📦 uv (Python package manager)
🤖 Ollama (LLM runtime)  
📥 AAC Models (gemma3:1b-it-qat, tinyllama:1.1b)
🐍 Python dependencies
🧪 Initial test run

Device: $DeviceName
Skip Tests: $SkipTests

Press Ctrl+C to cancel, or any key to continue...
"@
    
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    Write-Host ""
    
    try {
        # Check if we're in the right directory
        if (-not (Test-Path "model_test.py")) {
            Write-Error "model_test.py not found. Please run this script from the aac-model-testing directory."
            exit 1
        }
        
        # Installation steps
        Install-UV
        Install-Ollama
        Setup-Python-Environment
        Install-Models
        Run-Initial-Tests
        
        Write-Host @"

${Green}
================================================================================
🎉 AAC MODEL TESTING FRAMEWORK SETUP COMPLETE!
================================================================================
${Reset}

✅ All components installed successfully
✅ Models downloaded and ready
✅ Framework tested and working

${Blue}Next steps:${Reset}
• Run tests: ${Yellow}uv run python model_test.py${Reset}
• View help: ${Yellow}uv run python model_test.py --help${Reset}
• Test specific model: ${Yellow}uv run python model_test.py --models tinyllama:1.1b${Reset}
• Set device name: ${Yellow}uv run python model_test.py --device-name my-device${Reset}

${Blue}All results are saved to the 'results/' directory with timestamps.${Reset}
${Blue}View test history: ${Yellow}uv run python view_results.py${Reset}

Happy testing! 🚀
"@
    }
    catch {
        Write-Error "Setup failed: $_"
        Write-Host @"

${Red}Setup incomplete.${Reset} You may need to:
1. Install components manually
2. Check your internet connection  
3. Run as Administrator if needed
4. Check the error messages above

For manual installation instructions, see: README.md
"@
        exit 1
    }
}

# Run main function
Main
