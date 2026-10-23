# Portable GPU regression fixtures

These are byte-exact copies of the 100 existing remote cases: the unchanged
default 93 plus seven campaign witnesses. No new cases or tolerances were added
by this export. Each case includes typed IR, launch/ABI metadata, inputs,
independent expected outputs and its original case record.

From the repository root:

```sh
python -m tools.bundled_regressions
python -m tools.bundled_regressions --suite campaign100 --out /tmp/emmy-catalog.json
python -m tools.regression_suite --catalog /tmp/emmy-catalog.json --out /tmp/emmy-default93.json
```

Use fresh output filenames. The first command verifies all 100 payloads and the
original 93-case policy. The other commands produce machine-local absolute paths;
the checked-in manifest uses relative paths and works after moving the checkout.
Historical paths inside original case provenance are descriptive, not runtime
dependencies. This directory is not Git-ignored.

Hash verification does not execute GPU tests. Existing preparation and GPU tools
still require their Linux/CUDA runtime and dependencies. The manifest identifies
the remote source and correctness receipt associated with the export; those are
historical evidence, not a fresh pass for an edited compiler or another GPU.
