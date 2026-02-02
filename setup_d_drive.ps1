# Configure pip to use D: drive permanently
# Run this in PowerShell to set environment variables for current session

$env:PIP_CACHE_DIR="D:\pip_cache"
$env:TMPDIR="D:\pip_temp"
$env:TEMP="D:\pip_temp"
$env:TMP="D:\pip_temp"

Write-Host "✅ Pip configured to use D: drive for cache and temp files"
Write-Host "PIP_CACHE_DIR: $env:PIP_CACHE_DIR"
Write-Host "TEMP: $env:TEMP"
Write-Host ""
Write-Host "Now you can run: pip install -r requirements-minimal.txt"
