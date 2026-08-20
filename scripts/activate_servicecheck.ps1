[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

$projectRoot = [System.IO.Path]::GetFullPath(
    (Join-Path -Path $PSScriptRoot -ChildPath "..")
)

if ([System.IO.Path]::GetPathRoot($projectRoot) -ieq "C:\") {
    throw "The serviceCheck project root must not be on C:. Resolved path: $projectRoot"
}

$runtimeDirectories = @(
    "$projectRoot\.runtime\tmp",
    "$projectRoot\.runtime\pytest",
    "$projectRoot\.runtime\pip-cache",
    "$projectRoot\.runtime\conda-pkgs",
    "$projectRoot\.runtime\generic-cache",
    "$projectRoot\.runtime\postgres"
)

$persistentDirectories = @(
    "$projectRoot\model_store\modelscope",
    "$projectRoot\model_store\huggingface\hub",
    "$projectRoot\model_store\sentence-transformers",
    "$projectRoot\model_store\torch",
    "$projectRoot\data\artifacts"
)

foreach ($directory in $runtimeDirectories + $persistentDirectories) {
    $resolvedDirectory = [System.IO.Path]::GetFullPath($directory)
    if (-not $resolvedDirectory.StartsWith(
        $projectRoot,
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
        throw "Refusing to create a directory outside the project root: $resolvedDirectory"
    }
    New-Item -ItemType Directory -Force -Path $resolvedDirectory | Out-Null
}

function Set-ServiceCheckVariables {
    $pythonPathEntries = @(
        $env:PYTHONPATH -split [System.IO.Path]::PathSeparator |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    )
    if ($pythonPathEntries -notcontains $projectRoot) {
        $env:PYTHONPATH = (@($projectRoot) + $pythonPathEntries) -join (
            [System.IO.Path]::PathSeparator
        )
    }

    $env:TEMP = "$projectRoot\.runtime\tmp"
    $env:TMP = "$projectRoot\.runtime\tmp"
    $env:PIP_CACHE_DIR = "$projectRoot\.runtime\pip-cache"
    $env:CONDA_PKGS_DIRS = "$projectRoot\.runtime\conda-pkgs"
    $env:XDG_CACHE_HOME = "$projectRoot\.runtime\generic-cache"

    $env:MODELSCOPE_CACHE = "$projectRoot\model_store\modelscope"
    $env:HF_HOME = "$projectRoot\model_store\huggingface"
    $env:HF_HUB_CACHE = "$projectRoot\model_store\huggingface\hub"
    $env:SENTENCE_TRANSFORMERS_HOME = "$projectRoot\model_store\sentence-transformers"
    $env:TORCH_HOME = "$projectRoot\model_store\torch"

    $env:DATABASE_URL = "postgresql+psycopg://servicecheck@127.0.0.1:55432/servicecheck"
    $env:ARTIFACT_ROOT = "$projectRoot\data\artifacts"
    $env:API_CORS_ORIGINS = "http://127.0.0.1:8080,http://localhost:8080"

    # This development machine has 16 GB RAM. Avoid keeping the FunASR model
    # family and emotion2vec resident at the same time, and bound native
    # OpenMP/BLAS threads inside the isolated emotion process.
    $env:BATCH_LOW_MEMORY_MODE = "1"
    $env:ASR_SUBPROCESS_NUM_THREADS = "2"
    $env:ASR_SUBPROCESS_TIMEOUT_SECONDS = "300"
    $env:EMOTION_SUBPROCESS_NUM_THREADS = "1"
    $env:EMOTION_SUBPROCESS_TIMEOUT_SECONDS = "300"
    $env:EMOTION_MAX_CHUNK_SECONDS = "30"

    # Python 3.14's platform.machine() prefers WMI. The local WMI provider can
    # block indefinitely; use Python's built-in registry/sysinfo fallback.
    $env:SERVICECHECK_DISABLE_PYTHON_WMI = "1"

    # Conda 24.11.3 may emit GBK when it renders Chinese paths for PowerShell.
    # Set UTF-8 before activation so PowerShell receives those paths intact.
    $env:PYTHONUTF8 = "1"
    $env:PYTHONIOENCODING = "utf-8"
}

Set-ServiceCheckVariables

$condaFunction = Get-Command -Name conda -CommandType Function -ErrorAction SilentlyContinue
if (-not $condaFunction) {
    $condaExecutable = Get-Command -Name conda.exe -CommandType Application -ErrorAction Stop
    $condaRoot = Split-Path -Parent (Split-Path -Parent $condaExecutable.Source)
    $condaModule = Join-Path $condaRoot "shell\condabin\Conda.psm1"

    if (-not (Test-Path -LiteralPath $condaModule -PathType Leaf)) {
        throw "Conda PowerShell module was not found: $condaModule"
    }

    $env:CONDA_EXE = $condaExecutable.Source
    Remove-Item Env:_CE_M -ErrorAction SilentlyContinue
    Remove-Item Env:_CE_CONDA -ErrorAction SilentlyContinue
    Import-Module $condaModule -ArgumentList @{ ChangePs1 = $true }
}

conda activate servicecheck

# Reapply project-derived paths after activation. This keeps the script correct
# if the project directory is moved after Conda's persisted values were created.
Set-ServiceCheckVariables

$expectedPython = Join-Path $env:CONDA_PREFIX "python.exe"
if ((Split-Path -Leaf $env:CONDA_PREFIX) -cne "servicecheck") {
    throw "Unexpected active Conda environment: $env:CONDA_PREFIX"
}
if (-not (Test-Path -LiteralPath $expectedPython -PathType Leaf)) {
    throw "The servicecheck Python interpreter was not found: $expectedPython"
}

Set-Location -LiteralPath $projectRoot

Write-Host "servicecheck environment activated."
Write-Host "Python: $expectedPython"
Write-Host "Project: $projectRoot"
Write-Host "Database: $env:DATABASE_URL"
