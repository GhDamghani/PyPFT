# Builds the sdist and wheel into a clean dist/ and checks their metadata renders
# on PyPI. With -Tag (a release tag such as v0.14.0), also fails unless the tag
# names the version in pyproject.toml, so a release can never upload a
# distribution whose version differs from its tag.

param(
    [string]$Tag
)

$ErrorActionPreference = "Stop"

$DistDirectory = "dist"

if ($Tag) {
    $version = (uv version --short).Trim()
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
    if ($Tag -ne "v$version") {
        Write-Error "Release tag '$Tag' does not match the package version 'v$version'."
        exit 1
    }
}

if (Test-Path $DistDirectory) {
    Remove-Item -Recurse -Force $DistDirectory
}

uv build --out-dir $DistDirectory
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

uvx twine check --strict "$DistDirectory/*"
exit $LASTEXITCODE
