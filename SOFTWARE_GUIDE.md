# CLASSIX Software Guide

This guide describes the software architecture and development workflow of CLASSIX. It complements the user documentation by explaining how the estimator, metric-specific algorithms, compiled extensions, tests, and release automation fit together.

## 1. Scope

CLASSIX is a Python package for clustering dense numerical data with Euclidean, Manhattan, or Tanimoto distance. The public entry point is `classix.CLASSIX`.

Supported inputs include NumPy arrays, array-like objects, and Pandas DataFrames. One-dimensional inputs are reshaped automatically. The current public interface does not perform missing-value imputation, categorical encoding, or domain-specific feature construction. Tanimoto distance is intended for non-negative binary, count, or fingerprint-like features.

CLASSIX explanations describe how aggregation groups and their representatives connect to form clusters. They are not causal explanations and are not feature-attribution methods.

## 2. Repository layout

The main software components are:

- `classix/clustering.py`: public estimator, input validation, preprocessing, backend dispatch, fitted state, prediction, explanation, diagnostics, and plotting.
- `classix/aggregate_ed.py` and `classix/merge_ed.py`: portable Euclidean aggregation and merging implementations.
- `classix/aggregate_ed_c.pyx`, `classix/aggregate_ed_cm.pyx`, `classix/merge_ed_cm.pyx`, and `classix/merge_ed_cm_win.pyx`: compiled Euclidean implementations.
- `classix/aggregate_md.py` and `classix/merge_md.py`: portable Manhattan implementations.
- `classix/aggregate_md_cm.pyx` and `classix/merge_md_cm.pyx`: compiled Manhattan implementations.
- `classix/aggregate_td.py` and `classix/merge_td.py`: portable Tanimoto implementations.
- `classix/aggregate_td_cm.pyx` and `classix/merge_td_cm.pyx`: compiled Tanimoto implementations.
- `classix/spmv.cpp`: sparse matrix--vector kernel used by the Tanimoto implementation.
- `unittests.py`: regression and coverage tests.
- `docs/source/`: Sphinx user documentation.
- `demos/`: user-facing examples and notebooks.
- `exps/`: research and benchmarking material that is not imported by the package.
- `.github/workflows/codecov.yml`: compiled test and coverage workflow.
- `.github/workflows/wheels.yml`: tagged-release wheel build and PyPI publishing workflow.

File names may evolve. The invariant is the separation between public orchestration, metric-specific aggregation/merging, compiled acceleration, tests, documentation, and research experiments.

## 3. Public estimator lifecycle

A `CLASSIX` object follows this lifecycle:

1. Construction validates high-level parameters, selects a metric, and detects whether compiled extensions are importable.
2. `fit(data)` converts and validates the input, applies metric-specific preprocessing, aggregates sorted samples into groups, merges groups into clusters, restores original input order, and stores fitted state.
3. `fit_transform(data)` calls `fit` and returns `labels_`.
4. `predict(data)` assigns new samples to the cluster of the nearest fitted group representative. This is a nearest-representative rule, not a refit.
5. `explain(...)` reports global information, explains one sample, or reports a representative path between two samples.
6. `getPath(...)` exposes the representative path programmatically.

Important fitted attributes include:

- `labels_`: final labels in original input order.
- `groups_`: aggregation-group labels in sorted order.
- `splist_`: representative or starting-point information.
- `ind` and `inverse_ind`: mappings between original and sorted order.
- `clusterSizes_` and `groupCenters_`: derived fitted summaries.
- `mu_` and `dataScale_`: preprocessing state.
- `nrDistComp_`: number of distance evaluations recorded by aggregation.
- timing attributes for preprocessing, aggregation, merging, and small-cluster handling.

## 4. Backend contract

The estimator expects each metric backend to expose compatible information even when its internal representation differs.

### 4.1 Aggregation output

An aggregation backend must provide:

- one aggregation-group label for each sample in sorted order;
- representative indices (`splist` or equivalent);
- the permutation mapping original input to sorted order;
- sorting values in sorted order;
- the sorted data used by merging and explanation;
- aggregation-group sizes where required;
- the number of evaluated distances.

The estimator must be able to construct final labels and representative coordinates from these outputs.

### 4.2 Merging output

A merging backend must:

- accept aggregation labels and representative information;
- apply the selected group-merging rule;
- support `minPts` handling consistently with the estimator contract;
- return final labels that can be mapped back to original input order;
- preserve enough connectivity information for explanations where the metric supports it.

### 4.3 Compiled and Python implementations

Compiled and portable implementations should be semantically equivalent. Adding an optimization must not change the public estimator signature or fitted-attribute meanings. Tests should compare labels or invariant summaries on deterministic inputs.

When compiled extensions cannot be imported, the package should issue a clear warning and use the portable implementation. `classix.cython_is_available()` reports the active availability state. Developers can set `classix.__enable_cython__ = False` before constructing an estimator to exercise fallback paths.

## 5. Local development

Create an isolated environment, install build requirements, build extensions, and install the package in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
python setup.py build_ext --inplace
```

A minimal smoke test is:

```bash
python - <<'PY'
import numpy as np
import classix

X = np.random.default_rng(0).normal(size=(200, 4))
model = classix.CLASSIX(radius=0.5, verbose=0).fit(X)
print(classix.__version__)
print(model.labels_.shape, model.nrDistComp_)
print("Cython available:", classix.cython_is_available())
PY
```

## 6. Testing

Install test dependencies and run the repository test suite:

```bash
python -m pip install pytest pytest-cov coverage
pytest unittests.py
```

To reproduce the coverage workflow:

```bash
find . -name "*.so" -delete
find . -name "*.c" -delete
rm -rf build dist *.egg-info
CFLAGS="-O0 -DCYTHON_TRACE=1" python setup.py build_ext --inplace
python -m pip install -e .
pytest --cov=classix --cov-report=term-missing --cov-report=xml:coverage.xml unittests.py
```

On Windows, use the equivalent PowerShell cleanup commands from `.github/workflows/codecov.yml`.

Every behavior change should include a regression test. Tests should cover both NumPy and DataFrame inputs when index behavior matters, pre-fit errors for fitted methods, deterministic prediction, explanation paths, plotting without an interactive display, and compiled/fallback selection where practical.

## 7. Building documentation

Install Sphinx and the documentation requirements used by the project, then run:

```bash
cd docs
make html
```

The generated site is written to `docs/build/html/`. Public methods and attributes should have NumPy-style docstrings. User-facing behavior changes should update the API reference and at least one example.

## 8. Adding a metric

A new metric should be added only when its sorting/pruning rule and merging behavior are clearly defined.

1. Add a portable aggregation module and, if applicable, a portable merging module.
2. Make the modules return the aggregation and merging contract described above.
3. Add compiled implementations only after the portable version is tested.
4. Register the metric and its validation rules in `CLASSIX.__init__` and `fit`.
5. Define preprocessing, prediction distance, and explanation behavior.
6. Add deterministic unit tests for labels, group information, prediction, invalid inputs, and fallback behavior.
7. Add API documentation and an example that explains when the metric is appropriate.
8. Confirm wheel builds on Linux, Windows, and macOS before release.

Metric-specific restrictions must be explicit. For example, a similarity derived from non-negative fingerprints should reject or clearly document unsupported negative inputs.

## 9. Adding an example or notebook

Examples should be reproducible, reasonably small, and licensed for redistribution. A contribution should include:

- a short statement of the use case;
- the data source and license;
- pinned random seeds;
- installation requirements beyond the base package;
- expected runtime and approximate memory use;
- a textual explanation query, not only a cluster plot;
- no large generated outputs committed to Git.

Place polished user examples in `demos/` or the Sphinx examples page. Place paper-specific benchmark scripts in `exps/`.

## 10. Release checklist

Before tagging a release:

1. Update `classix.__version__`.
2. Run the complete test suite with compiled extensions.
3. Exercise the Python fallback on representative Euclidean, Manhattan, and Tanimoto inputs.
4. Build the documentation and check links.
5. Confirm that packaging metadata and supported Python versions are correct.
6. Tag the release with `v<version>` to trigger `.github/workflows/wheels.yml`.
7. Verify wheel artifacts and the PyPI release.
8. Record the immutable release tag or commit in associated papers and archival material.

## 11. Reporting issues

Bug reports should include the CLASSIX version, Python version, operating system, whether Cython is available, a minimal input or reproduction script, the expected behavior, and the observed traceback or output. Performance reports should also include input shape, dtype, metric, parameters, runtime, and `nrDistComp_`.
