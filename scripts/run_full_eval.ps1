param(
    [int]$Limit = 120,
    [string[]]$Models = @("qwen2.5:7b", "llama3.1:8b", "qwen2.5:3b"),
    [string]$ModelDir = "D:\JZDSTool\ollama_models",
    [string]$CacheDir = "D:\JZDSTool\nlp_course_cache",
    [int]$NumPredict = 256,
    [int]$NumCtx = 4096,
    [int]$Step1NumPredict = 192,
    [int]$Step2NumPredict = 256,
    [int]$Step3NumPredict = 384,
    [int]$Step4NumPredict = 160
)

$ErrorActionPreference = "Stop"
$env:OLLAMA_MODELS = $ModelDir
New-Item -ItemType Directory -Force -Path $ModelDir | Out-Null
New-Item -ItemType Directory -Force -Path $CacheDir | Out-Null
$env:HF_HOME = Join-Path $CacheDir "hf_home"
$env:HF_DATASETS_CACHE = Join-Path $CacheDir "hf_datasets"
$env:TRANSFORMERS_CACHE = Join-Path $CacheDir "transformers"
$env:TMP = Join-Path $CacheDir "tmp"
$env:TEMP = Join-Path $CacheDir "tmp"
$env:OLLAMA_NUM_PREDICT = [string]$NumPredict
$env:OLLAMA_NUM_CTX = [string]$NumCtx
$env:OLLAMA_STEP1_NUM_PREDICT = [string]$Step1NumPredict
$env:OLLAMA_STEP2_NUM_PREDICT = [string]$Step2NumPredict
$env:OLLAMA_STEP3_NUM_PREDICT = [string]$Step3NumPredict
$env:OLLAMA_STEP4_NUM_PREDICT = [string]$Step4NumPredict
New-Item -ItemType Directory -Force -Path $env:HF_HOME, $env:HF_DATASETS_CACHE, $env:TRANSFORMERS_CACHE, $env:TMP | Out-Null
$projectRoot = Split-Path $PSScriptRoot -Parent
Set-Location $projectRoot

$ollamaExe = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
if (Test-Path $ollamaExe) {
    $ollamaDir = Split-Path $ollamaExe
    if (($env:Path -split ";") -notcontains $ollamaDir) {
        $env:Path = "$ollamaDir;$env:Path"
    }
}

python -m app.cli check-ollama
python -m app.cli evaluate --models $Models --limit $Limit --mode one-shot --dataset public-aste
python -m app.cli evaluate --models $Models --limit $Limit --mode four-step --dataset public-aste
python -m app.cli evaluate-flips --model $($Models[0]) --mode four-step
