# Runs PyPFT's quality gate: the checks every phase must leave green. Assumes `uv sync`
# has already run.

$ErrorActionPreference = "Stop"

function Invoke-Step {
    param(
        [Parameter(Mandatory)][string]$Description,
        [Parameter(Mandatory)][string[]]$Command
    )
    Write-Host "==> $Description"
    & $Command[0] $Command[1..($Command.Length - 1)]
    if ($LASTEXITCODE -ne 0) {
        Write-Error "$Description failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }
}

Invoke-Step "pytest" @("uv", "run", "pytest")
Invoke-Step "notebooks (nbmake)" @("pwsh", "-NoProfile", "-File", "scripts/Test-Notebooks.ps1")
Invoke-Step "black --check" @("uv", "run", "black", "--check", "src", "tests", "benchmarks", "scripts")
Invoke-Step "isort --check-only" @("uv", "run", "isort", "--check-only", "src", "tests", "benchmarks", "scripts")
Invoke-Step "flake8" @("uv", "run", "flake8", "src", "scripts")
Invoke-Step "pyright" @("uv", "run", "pyright")
Invoke-Step "vulture" @("uv", "run", "vulture", "src", "scripts")
Invoke-Step "sphinx-build -W" @("uv", "run", "sphinx-build", "-W", "docs", "docs/_build")
Invoke-Step "uv build" @("uv", "build")
