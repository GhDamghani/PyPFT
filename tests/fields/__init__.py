"""Field-layer tests use fixtures inherited from tests/conftest.py.

Pytest loads parent ``conftest.py`` files for descendants under the configured
``tests`` tree, so ``sample_image`` and ``sample_batch`` remain available in
this subdirectory without a local fixture module.
"""