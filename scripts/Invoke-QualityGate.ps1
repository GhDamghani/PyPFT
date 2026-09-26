# Runs PyPFT's quality gate: the checks every phase must leave green. Assumes `uv sync`
# has already run.
#
# -NoSync runs every step with `uv run --no-sync`, so an environment deliberately
# upgraded past the locked dependency floors (CI's "latest" leg) is tested as-is
# instead of being re-synced back to uv.lock first.

param(
    [switch]$NoSync
)

$ErrorActionPreference = "Stop"

# Every `uv run` below goes through this prefix, so -NoSync covers all of them.
$UvRun = @("uv", "run")
if ($NoSync) {
    $UvRun += "--no-sync"
}

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

# Forward -NoSync to the notebook script, which runs its own `uv run`.
$NotebookCommand = @("pwsh", "-NoProfile", "-File", "scripts/Test-Notebooks.ps1")
if ($NoSync) {
    $NotebookCommand += "-NoSync"
}

Invoke-Step "pytest" ($UvRun + @("pytest"))
Invoke-Step "notebooks (nbmake)" $NotebookCommand
Invoke-Step "black --check" ($UvRun + @("black", "--check", "src", "tests", "benchmarks", "scripts"))
Invoke-Step "isort --check-only" ($UvRun + @("isort", "--check-only", "src", "tests", "benchmarks", "scripts"))
Invoke-Step "flake8" ($UvRun + @("flake8", "src", "scripts"))
Invoke-Step "pyright" ($UvRun + @("pyright"))
Invoke-Step "vulture" ($UvRun + @("vulture", "src", "scripts"))
Invoke-Step "sphinx-build -W" ($UvRun + @("sphinx-build", "-W", "docs", "docs/_build"))
Invoke-Step "uv build" @("uv", "build")
