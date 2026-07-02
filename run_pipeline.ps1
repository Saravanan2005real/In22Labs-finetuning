# LexAI Legal Chatbot Pipeline Runner

param(
    [parameter(Mandatory=$false)]
    [ValidateSet("extract", "train")]
    [string]$Action = "extract"
)

# Ensure virtual environment exists
if (-not (Test-Path "finetune_env\Scripts\activate.ps1")) {
    Write-Host "ERROR: finetune_env virtual environment not found. Please wait for environment setup to complete." -ForegroundColor Red
    exit 1
}

# Activate environment
. finetune_env\Scripts\Activate.ps1

switch ($Action) {
    "extract" {
        Write-Host "=== STEP 1: Extracting text from PDFs and converting to JSON ===" -ForegroundColor Cyan
        python extract_to_json.py
    }
    
    "train" {
        Write-Host "=== STEP 2: Fine-tuning the Llama model on the extracted dataset ===" -ForegroundColor Cyan
        python fine_tune.py
    }
}
