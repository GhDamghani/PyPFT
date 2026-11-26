# Security policy

## Supported versions

Security fixes go into the latest release on PyPI only. Upgrade with `pip install --upgrade pypft`.

## Reporting a vulnerability

Please report vulnerabilities privately, not in a public issue: use GitHub's [private vulnerability reporting](https://github.com/GhDamghani/PyPFT/security/advisories/new) for this repository.

Include what you found, how to reproduce it, and which version you used. You can expect a first reply within 14 days. Once a fix is released, the advisory is published with credit to you, unless you prefer otherwise.

## Scope

PyPFT is a numerical library: it reads array files you give it and, in `pypft.samples`, downloads test data from published URLs and checks each file's MD5 checksum. Reports about either, or about any way a crafted input runs code or reads or writes files outside what was asked, are in scope.
