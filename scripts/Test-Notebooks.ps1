# Executes every notebook under notebooks/ via nbmake, so tutorial drift fails CI
# rather than rotting silently (SS2.3 of .local_files/develop_plan.md). Notebooks land
# starting in P3; until then this collects zero items, which pytest reports as exit
# code 5 ("no tests collected") -- treated here as a pass, not a failure.

$ErrorActionPreference = "Stop"

uv run pytest --nbmake notebooks/
$exitCode = $LASTEXITCODE

if ($exitCode -eq 5) {
    Write-Host "No notebooks collected yet; nothing to execute."
    exit 0
}

exit $exitCode
