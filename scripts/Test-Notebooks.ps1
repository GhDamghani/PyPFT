# Executes every notebook under notebooks/ via nbmake, so tutorial drift fails CI
# rather than rotting silently. If notebooks/ is ever empty, pytest reports exit
# code 5 ("no tests collected") -- treated here as a pass, not a failure.
#
# -NoSync runs with `uv run --no-sync`, leaving an upgraded environment untouched
# (see scripts/Invoke-QualityGate.ps1).

param(
    [switch]$NoSync
)

$ErrorActionPreference = "Stop"

$UvRun = @("run")
if ($NoSync) {
    $UvRun += "--no-sync"
}

uv @UvRun pytest --nbmake notebooks/
$exitCode = $LASTEXITCODE

if ($exitCode -eq 5) {
    Write-Host "No notebooks collected yet; nothing to execute."
    exit 0
}

exit $exitCode
