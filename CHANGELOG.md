# Changelog

All notable changes to pydreg are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
uses [Semantic Versioning](https://semver.org/) (pre-1.0: a minor bump marks a
change to default output or the public interface).

Performance changes are summarized here; the full measurements, including
approaches that were tried and rejected, are in
[`docs/PERF_LOG.md`](docs/PERF_LOG.md).

## [Unreleased]

### Added
- This changelog.
- A "Project status" section in the README covering maintenance mode, what is
  out of scope, and where contributions help.
- A `Changelog` link in the package metadata, shown in the PyPI sidebar.

### Changed
- The README's caveat about p-value noise now points to `--seed` for
  reproducible peak calling.
- GPU (`cupy`) scoring is about 1.2x faster: the RBF step and the
  multiplication by the SVR coefficients now run as one fused kernel, so the
  full kernel matrix is never written to GPU memory. Scores move by at most
  ~1e-3 (float32 noise) and are closer to the float64 reference than before;
  peak agreement with real dREG on Jurkat_ChROseq_1 went from 0.999552 to
  0.999764 Jaccard.

## [0.3.4] - 2026-11-01

### Changed
- Locked figwig at 0.2.0 (the `figwig>=0.1` floor is unchanged). The optional
  ISA-L decoder was benchmarked and not adopted because it was slightly slower
  than zlib on real data.
- Feature extraction is about 31% faster: a single-pass numba prefix-sum kernel
  replaces four full-size array copies per cluster strand, and the minus
  strand's prefix sum now runs alongside the plus strand's. Features are
  bit-identical to before.
- `fetch_raw` now limits reads to the chromosome instead of NaN-cleaning every
  base.

### Fixed
- Infinite values in input bigWigs now raise `ValueError` instead of being
  silently turned into ±1.8e308.
- `CITATION.cff` version was stale (still 0.3.2).

## [0.3.3] - 2026-10-04

### Changed
- Replaced pybigtools with [figwig](https://pypi.org/project/figwig/) for all
  bigWig reading and writing. This adds Python 3.14 support, removes a
  thread-safety crash, and removes the Rust toolchain requirement that broke
  installs on some HPC clusters.
- Plus- and minus-strand reads now overlap during informative-position scanning
  and feature extraction.
- `windowed_sum` reads in 5 Mbp chunks to limit transient memory.

### Fixed
- The cupy-not-installed test failed on machines that have a GPU.

### Added
- pytest-cov dev dependency, coverage badge, and `CONTRIBUTING.md`.

## [0.3.2] - 2026-09-18

### Added
- `--seed` (`pipeline.run(seed=)`, `call_peaks(seed=)`) makes peak calling
  reproducible, including across different `--cores` values. Runs are still
  unseeded by default.

### Changed
- pydreg no longer monkeypatches SciPy's private QMC module globals. It now
  calls SciPy's compiled kernel through its own driver, with an import-time
  signature check that catches changes to that private API. Output is
  bit-identical.

## [0.3.1.post2] - 2026-09-13

### Added
- Citation for the pydreg preprint.
- GPU profiling write-up explaining why dREG's GPU path is slower, measured
  across all 12 benchmark libraries.

## [0.3.1.post1] - 2026-08-28

### Fixed
- Stale MIT license classifier in package metadata.

## [0.3.1] - 2026-08-23

### Fixed
- Removed a misleading "cupy missing" log message on machines with no NVIDIA
  GPU, such as Apple silicon.

## [0.3.0] - 2026-08-23

### Changed
- **Default peak-calling output changed:** the bounded tail-truncation fast mode
  for `pmv_laplace` (`tail_tol=1e-6`) is now the default. Validation against
  real dREG gave identical peak counts and the same Jaccard agreement.
  `--pmv-laplace-fast` is replaced by `--pmv-laplace-exact`, which restores the
  exact integration. `--pmv-laplace-tail-tol` still overrides either setting.
- Lowered the default `sv_chunk` of the cupy and mlx backends from 32,768 to
  16,384.

## [0.2.7.post1] - 2026-08-17

### Fixed
- Broken benchmark image link in the PyPI project description. No code changes.

## [0.2.7] - 2026-08-15

### Added
- mlx scoring backend for Apple silicon GPUs.

### Changed
- Feature extraction and informative-position scanning are numba-compiled.
- A single `--cores` value now sets peak-calling processes, numba threads, and
  BLAS threads.
- CPU scoring's RBF-kernel step is fused into a parallel numba kernel.
- Density-aware clustering for feature extraction, and parallel output writing.
- Model loading is faster and is logged as its own pipeline stage.
- Fixed bigWig write-thread contention.

### Removed
- The debug-only informative-position `bed.gz` output.

## [0.2.6] - 2026-07-25

### Added
- `--svr-model` and `--rf-model` load model files from local paths instead of
  downloading them from Hugging Face.

## [0.2.5] - 2026-07-21

### Changed
- Faster peak calling: `pmv_laplace` skips z-grid evaluations that are provably
  zero.

## [0.2.4] - 2026-07-17

### Fixed
- Relaxed the tolerance of the CuPy smoke test.

## [0.2.3] - 2026-07-16

### Added
- `dreg` and `dREG` command aliases for `pydreg`.

### Fixed
- Contigs present in only one strand's bigWig are now read as all-zero signal
  instead of raising an error.
- Guarded the model download cache.

## [0.2.2] - 2026-07-15

### Fixed
- The `gpu` extra now installs `cupy-cuda12x[ctk]` so the CUDA toolkit
  components CuPy needs are present.

## [0.2.1] - 2026-07-15

### Changed
- Lowered the minimum Python version to 3.11; CI tests through 3.13.

## [0.2.0] - 2026-07-15

### Added
- A custom CuPy GPU scoring backend.

### Removed
- The cuML backend, which returned wrong scores below CUDA compute
  capability 7.0.

## [0.1.2] - 2026-07-15

### Fixed
- Enforced CUDA compute capability >= 7.0 for the cuML GPU backend.

## [0.1.1] - 2026-07-14

### Fixed
- Documentation text corrections.

## [0.1.0] - 2026-07-14

### Added
- First public release: an end-to-end Python port of dREG's peak-calling
  workflow (informative-position scanning, feature extraction, SVR scoring,
  RF-assisted peak calling, FDR filtering, output writing), with the
  pretrained models hosted on Hugging Face.

[Unreleased]: https://github.com/adamyhe/pydreg/compare/v0.3.4...HEAD
[0.3.4]: https://github.com/adamyhe/pydreg/compare/v0.3.3...v0.3.4
[0.3.3]: https://github.com/adamyhe/pydreg/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/adamyhe/pydreg/compare/v0.3.1.post2...v0.3.2
[0.3.1.post2]: https://github.com/adamyhe/pydreg/compare/v0.3.1.post1...v0.3.1.post2
[0.3.1.post1]: https://github.com/adamyhe/pydreg/compare/v0.3.1...v0.3.1.post1
[0.3.1]: https://github.com/adamyhe/pydreg/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/adamyhe/pydreg/compare/v0.2.7.post1...v0.3.0
[0.2.7.post1]: https://github.com/adamyhe/pydreg/compare/v0.2.7...v0.2.7.post1
[0.2.7]: https://github.com/adamyhe/pydreg/compare/v0.2.6...v0.2.7
[0.2.6]: https://github.com/adamyhe/pydreg/compare/v0.2.5...v0.2.6
[0.2.5]: https://github.com/adamyhe/pydreg/compare/v0.2.4...v0.2.5
[0.2.4]: https://github.com/adamyhe/pydreg/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/adamyhe/pydreg/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/adamyhe/pydreg/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/adamyhe/pydreg/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/adamyhe/pydreg/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/adamyhe/pydreg/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/adamyhe/pydreg/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/adamyhe/pydreg/releases/tag/v0.1.0
