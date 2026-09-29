<#
.SYNOPSIS
    Report the CPython versions PyPFT is tested on to a GitHub Actions workflow.

.DESCRIPTION
    Reads supported_python_versions.txt (repo root) -- the single place the
    supported versions are written down -- so .github/workflows/ci.yml builds its
    matrix from that file instead of repeating the list in YAML. Blank lines and
    "#" comments are ignored.

    Writes the list to $env:GITHUB_OUTPUT as "versions", a JSON array (the only
    form a matrix can consume), when that variable is set, and always emits the
    versions to the pipeline so the script is also useful locally.

.EXAMPLE
    pwsh scripts/Get-PythonVersions.ps1
#>
[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

$versionsFile = Join-Path (Split-Path $PSScriptRoot -Parent) "supported_python_versions.txt"

# Keep only non-blank lines that are not comments, trimmed.
$versions = @(
    Get-Content -LiteralPath $versionsFile |
        ForEach-Object { $_.Trim() } |
        Where-Object { $_ -and -not $_.StartsWith("#") }
)
if ($versions.Count -eq 0) {
    throw "No Python versions listed in $versionsFile"
}

# ConvertTo-Json renders a single-element array as a bare scalar unless forced
# with -AsArray, which a matrix would then reject.
$json = $versions | ConvertTo-Json -Compress -AsArray
if ($env:GITHUB_OUTPUT) {
    Add-Content -LiteralPath $env:GITHUB_OUTPUT -Value "versions=$json"
}

Write-Output $versions
