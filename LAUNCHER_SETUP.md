# Jarvis Desktop Launcher Setup

## Quick Start
Double-click `launcher.bat` to start Jarvis. It will:
1. Check if Ollama is running
2. Start Ollama if needed
3. Activate the Python venv
4. Launch Jarvis (opens browser at http://127.0.0.1:8765/)

## Create Desktop Shortcut (Manual)

### Option 1: Right-Click Shortcut Creation
1. Navigate to the Jarvis repository folder in File Explorer
2. Right-click on `launcher.bat`
3. Select **"Create shortcut"**
4. Move the shortcut to your Desktop
5. (Optional) Right-click the shortcut → **Properties** → change icon or name

### Option 2: Manual Shortcut Creation
1. Right-click on Desktop → **New** → **Shortcut**
2. For the location, paste:
   ```
   C:\Users\andre\source\repos\Jarvis\launcher.bat
   ```
   (Adjust path if your repo is elsewhere)
3. Name it: `Jarvis`
4. Click **Finish**
5. (Optional) Right-click shortcut → **Properties** → **Advanced** → check "Run as Administrator" (not required, but recommended)

### Option 3: Pin to Start Menu
Instead of desktop, you can:
1. Right-click `launcher.bat` in File Explorer
2. Select **"Pin to Start"** or **"Pin to Taskbar"**

## Files Created
- **launcher.ps1** - Main launcher script (checks Ollama, starts Jarvis)
- **launcher.bat** - Batch wrapper (handles PowerShell execution policy)
- **web/favicon.svg** - Jarvis icon for browser tab
- **LAUNCHER_SETUP.md** - This file

## Ollama Detection
The launcher checks if Ollama is running on `localhost:11434`:
- If running: Skips startup, goes straight to Jarvis
- If not running: Attempts to start Ollama from:
  - `%LOCALAPPDATA%\Programs\Ollama\ollama.exe`
  - `C:\Program Files\Ollama\ollama.exe`

## Troubleshooting

### "Ollama executable not found"
- Install Ollama from https://ollama.ai
- Restart the launcher after installation

### "Virtual environment not found"
```powershell
python -m venv .venv
pip install -r requirements.txt
```
Then run launcher again.

### Port 11434 already in use
- Another Ollama instance is running
- Kill it and restart, or just proceed (launcher will use the existing instance)

### Favicon not showing
- Hard-refresh browser: `Ctrl+Shift+Delete` or `Ctrl+F5`
- The favicon.svg should appear as a cyan circle on the browser tab

## Manual Launcher (if .bat doesn't work)
```powershell
cd C:\Users\andre\source\repos\Jarvis
.\.venv\Scripts\Activate.ps1
python run.py
```
(This assumes Ollama is already running or not needed)
