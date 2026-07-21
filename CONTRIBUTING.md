# Contributing to CLASSIX

Thank you for considering a contribution to CLASSIX. Contributions may include bug fixes, tests, documentation, performance improvements, new demonstrations, and carefully justified metric extensions.

## Before opening a pull request

1. Search existing issues and pull requests.
2. For a substantial API or algorithm change, open an issue describing the use case and proposed behavior.
3. Keep changes focused. Separate refactoring from behavior changes where possible.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
python setup.py build_ext --inplace
python -m pip install pytest pytest-cov coverage
pytest unittests.py
```

See `SOFTWARE_GUIDE.md` for architecture, backend contracts, documentation builds, and release details.

## Pull-request expectations

A pull request should:

- explain the problem and the chosen solution;
- include tests for behavior changes and bug fixes;
- preserve the public meanings of fitted attributes;
- document new parameters, methods, attributes, warnings, and limitations;
- avoid unrelated formatting changes;
- pass the compiled test suite;
- keep generated binaries, coverage files, and large notebook outputs out of the commit.

## Code and API conventions

- Use clear, descriptive names and NumPy-style docstrings.
- Validate public inputs early and raise informative exceptions.
- Keep metric-specific logic in metric-specific modules rather than adding deeply nested special cases to user-facing code.
- Maintain semantic equivalence between compiled and Python fallback implementations.
- Preserve original input ordering in final labels.
- Do not introduce a full pairwise distance matrix into the main fitting path without a documented reason.

## Documentation and demonstrations

Documentation fixes are welcome. New demonstrations should state the data source and license, set random seeds, note expected runtime, and include at least one explanation query. Domain-specific Jupyter notebooks are particularly useful, but clear outputs and large embedded data should be removed before committing.

## Bug reports

Please include:

- CLASSIX and Python versions;
- operating system;
- output of `classix.cython_is_available()`;
- a minimal reproducible example;
- expected and observed behavior;
- full error traceback, when applicable.

## Licensing

By contributing, you agree that your contribution will be distributed under the repository's MIT license.
