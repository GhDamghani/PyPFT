# Sets up a development copy of PyPFT in one step, from a fresh clone: syncs the
# uv environment, downloads the test samples too large to keep in the repository,
# and installs the pre-commit hook. Safe to re-run: every step skips work that is
# already done. Run `uv sync` on its own after a pull or branch switch.
#
# -SkipSamples leaves the ~230 MB sample download for later (the quality gate
# fetches missing samples itself).

param(
    [switch]$SkipSamples
)

$ErrorActionPreference = "Stop"

$SampleDirectory = "tests/samples"
$SampleCacheDirectory = ".local_files/samples"

function Invoke-Step {
    param(
        [string]$Name,
        [string[]]$Command
    )
    Write-Host "==> $Name"
    & $Command[0] $Command[1..($Command.Length - 1)]
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Step '$Name' failed with exit code $LASTEXITCODE."
        exit $LASTEXITCODE
    }
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Error "uv is not installed; see https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
}

Set-Location (Split-Path -Parent $PSScriptRoot)

Invoke-Step "uv sync" @("uv", "sync")
if (-not $SkipSamples) {
    Invoke-Step "download test samples" @(
        "uv", "run", "python", "-m", "pypft.samples",
        "--directory", $SampleDirectory,
        "--cache-directory", $SampleCacheDirectory
    )
}
Invoke-Step "install the pre-commit hook" @(
    "uv", "run", "pre-commit", "install", "--overwrite"
)

Write-Host "Done. Check the setup with ./scripts/Invoke-QualityGate.ps1"
