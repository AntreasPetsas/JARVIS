# Jarvis Launcher: Check Ollama and start the assistant
# Usage: .\launcher.ps1

$ErrorActionPreference = "Stop"

# Get the directory where this script is located
$scriptDir = Split-Path -Parent (Get-Item $PSCommandPath).FullName
Set-Location $scriptDir

# Track if we started Ollama (so we can clean it up)
$ollamaStartedByUs = $false

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "J.A.R.V.I.S Launcher" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Function to check if Ollama is running
function Test-OllamaRunning {
    try {
        $tcpClient = New-Object System.Net.Sockets.TcpClient
        $tcpClient.ConnectAsync("127.0.0.1", 11434).Wait(2000)
        if ($tcpClient.Connected) {
            $tcpClient.Close()
            return $true
        }
        return $false
    }
    catch {
        return $false
    }
}

# Function to unload all Ollama models
function Stop-OllamaModels {
    try {
        Write-Host "[*] Unloading Ollama models..." -ForegroundColor Yellow
        $response = Invoke-WebRequest -Uri "http://127.0.0.1:11434/api/generate" `
            -Method Post `
            -ContentType "application/json" `
            -Body '{"model":"","keep_alive":0}' `
            -ErrorAction SilentlyContinue
        Write-Host "[+] Ollama models unloaded" -ForegroundColor Green
    }
    catch {
        # If unload fails, that's okay - Ollama will handle it
    }
}

# Cleanup function for when Jarvis stops
function Cleanup {
    Write-Host ""
    Write-Host "[*] Cleaning up..." -ForegroundColor Yellow

    # Unload Ollama models
    Stop-OllamaModels

    # If we started Ollama, stop it
    if ($ollamaStartedByUs) {
        Write-Host "[*] Stopping Ollama..." -ForegroundColor Yellow
        try {
            Stop-Process -Name "ollama" -Force -ErrorAction SilentlyContinue
            Write-Host "[+] Ollama stopped" -ForegroundColor Green
        }
        catch {
            # Process might already be stopped
        }
    }

    Write-Host "[+] Cleanup complete" -ForegroundColor Green
}

# Register cleanup on exit (Ctrl+C, exit, or error)
$null = Register-EngineEvent -SourceIdentifier PowerShell.Exiting -Action { Cleanup }

# Check if Ollama is running
Write-Host "[*] Checking Ollama status..." -ForegroundColor Yellow
if (Test-OllamaRunning) {
    Write-Host "[+] Ollama is already running on port 11434" -ForegroundColor Green
    $ollamaStartedByUs = $false
}
else {
    Write-Host "[!] Ollama is not running, attempting to start..." -ForegroundColor Yellow

    # Try to start Ollama
    $ollamaExe = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
    $ollamaExeAlt = "C:\Program Files\Ollama\ollama.exe"

    if (Test-Path $ollamaExe) {
        Write-Host "[*] Found Ollama at: $ollamaExe" -ForegroundColor Cyan
        Start-Process $ollamaExe -WindowStyle Hidden
        $ollamaStartedByUs = $true
        Write-Host "[*] Ollama started in background..." -ForegroundColor Green
        Start-Sleep -Seconds 3
    }
    elseif (Test-Path $ollamaExeAlt) {
        Write-Host "[*] Found Ollama at: $ollamaExeAlt" -ForegroundColor Cyan
        Start-Process $ollamaExeAlt -WindowStyle Hidden
        $ollamaStartedByUs = $true
        Write-Host "[*] Ollama started in background..." -ForegroundColor Green
        Start-Sleep -Seconds 3
    }
    else {
        Write-Host "[!] Ollama executable not found. Please install Ollama from https://ollama.ai" -ForegroundColor Red
        Write-Host "[!] After installation, run this script again." -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }

    # Verify Ollama started
    if (-not (Test-OllamaRunning)) {
        Write-Host "[!] Ollama failed to start or is taking too long to initialize" -ForegroundColor Red
        Write-Host "[*] Please start Ollama manually and try again" -ForegroundColor Yellow
        Read-Host "Press Enter to exit"
        exit 1
    }
    Write-Host "[+] Ollama is now running" -ForegroundColor Green
}

Write-Host ""
Write-Host "[*] Activating virtual environment..." -ForegroundColor Yellow
$venvPath = Join-Path $scriptDir ".venv\Scripts\Activate.ps1"

if (-not (Test-Path $venvPath)) {
    Write-Host "[!] Virtual environment not found at: $venvPath" -ForegroundColor Red
    Write-Host "[!] Please run: python -m venv .venv" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

& $venvPath
Write-Host "[+] Virtual environment activated" -ForegroundColor Green

Write-Host ""
Write-Host "[*] Starting J.A.R.V.I.S..." -ForegroundColor Yellow
Write-Host "[*] Press Ctrl+C to stop" -ForegroundColor Cyan
Write-Host ""

# Run Jarvis
try {
    python run.py
}
catch {
    # Handle any errors
}
finally {
    # Always run cleanup
    Cleanup
}

Write-Host ""
Write-Host "[*] J.A.R.V.I.S session ended" -ForegroundColor Cyan
Read-Host "Press Enter to exit"
