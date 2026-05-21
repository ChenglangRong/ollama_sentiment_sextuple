param(
    [string[]]$Models = @("qwen2.5:7b", "llama3.1:8b", "qwen2.5:3b"),
    [string]$ModelDir = "D:\JZDSTool\ollama_models",
    [string]$CacheDir = "D:\JZDSTool\nlp_course_cache"
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

New-Item -ItemType Directory -Force -Path $ModelDir | Out-Null
New-Item -ItemType Directory -Force -Path $CacheDir | Out-Null
[Environment]::SetEnvironmentVariable("OLLAMA_MODELS", $ModelDir, "User")
$env:OLLAMA_MODELS = $ModelDir
$env:HF_HOME = Join-Path $CacheDir "hf_home"
$env:HF_DATASETS_CACHE = Join-Path $CacheDir "hf_datasets"
$env:TRANSFORMERS_CACHE = Join-Path $CacheDir "transformers"
$env:TMP = Join-Path $CacheDir "tmp"
$env:TEMP = Join-Path $CacheDir "tmp"
New-Item -ItemType Directory -Force -Path $env:HF_HOME, $env:HF_DATASETS_CACHE, $env:TRANSFORMERS_CACHE, $env:TMP | Out-Null

$ollamaExe = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
if (!(Test-Path $ollamaExe)) {
    Write-Host "Installing Ollama from official installer..."
    irm https://ollama.com/install.ps1 | iex
}

if (!(Test-Path $ollamaExe)) {
    throw "Ollama executable was not found after installation."
}

$ollamaDir = Split-Path $ollamaExe
if (($env:Path -split ";") -notcontains $ollamaDir) {
    $env:Path = "$ollamaDir;$env:Path"
}

Write-Host "Ollama executable: $ollamaExe"
Write-Host "Ollama model directory: $env:OLLAMA_MODELS"
& $ollamaExe list

foreach ($model in $Models) {
    Write-Host "Pulling $model ..."
    & $ollamaExe pull $model
}

Write-Host "Final model list:"
& $ollamaExe list
