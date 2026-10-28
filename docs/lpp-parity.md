# LPP Behavior and Parity Status

This document records the behavior implemented by the Python pipeline and separates that behavior from MATLAB parity evidence. The MATLAB source is the intended behavioral reference, but this repository currently has no redistributable MATLAB golden outputs or representative input image fixtures. Exact parity is therefore not established.

## Python Pipeline Contract

### Window Coordinates and Cropping

- Input is a two-dimensional grayscale array with shape `(height, width)`.
- Each `w x w` window starts at top-left coordinate `(row * dL, column * dL)`. Windows are enumerated in row-major order.
- A window is included only when its full extent fits inside the image. There is no padding at the image edge.
- The number of rows is `max(0, (height - w) // dL + 1)`; columns use the same formula with image width.
- `extract_windows_batched` bounds temporary window storage by `batch_size`. The public result still retains per-window metrics, so total memory is not independent of window count.

### Spectrum and Bins

- The pipeline subtracts each window mean, pads each window to eight times its width and height, computes a two-dimensional FFT, shifts the zero-frequency location to the center, and returns squared magnitude.
- Radial bins represent wavelengths in pixels over the configured `[minL, maxL]` interval. The default radial-bin count is `max(100, int(4 * (maxL - minL)))`.
- Angular bins use 36 director-angle bins over `[0, pi)`. Angular aggregation includes only frequencies in the requested radial wavelength interval and excludes DC.
- Empty radial or angular bins are represented by zero aggregate power. A spectrum with no positive aggregate power is an invalid window.

### Peak Metrics and Invalid Windows

- A radial maximum or angular maximum tied across multiple bins invalidates that window.
- A valid window's wavelength and orientation are power-weighted centroids around the radial and angular peak neighborhoods. Orientation averaging uses doubled angles, so directions separated by `pi` represent the same director.
- Direction is stored in radians in `[0, pi)`. It describes the dominant wavevector orientation from the FFT coordinates; interpreting it as stripe orientation may require a `pi/2` rotation depending on the downstream scientific convention.
- Uniqueness is the mean of the radial and angular peak-to-positive-bin-mean ratios. The `peak_ratio` configuration is the threshold.
- Invalid windows have `valid=False`, `wavelength=NaN`, and `direction=NaN`; their power and uniqueness arrays are initialized to zero. A valid window can still have `within_range=False` when its estimated wavelength falls outside the requested interval.

### Merge, Filters, and Edge Policy

- Windows contribute only when their peak is valid, wavelength is within range, and `powmax >= min_power`.
- Wavelength values are merged by pixel-wise sum/count. Directions are merged as directors using sums of `sin(2*theta)` and `cos(2*theta)`.
- `min_patch_size > 1` filters connected components of covered pixels. This operation requires SciPy; the pipeline now raises an actionable error if SciPy is missing instead of silently skipping the filter. Install `mcrd-accel[analysis]` to enable it, or use `min_patch_size=1` to disable component filtering.
- `edge_policy="trim"` (default) masks a half-window margin around the region reachable by complete windows. `"keep"` preserves the merged extent. `"nan"` masks a half-window margin around the full image boundary.
- LPP currently supports the NumPy backend only. Passing another backend name raises an error rather than silently running NumPy.

## Known Compatibility Gaps

- No MATLAB golden outputs or experimental image fixtures are currently checked in.
- The Python implementation uses eight-fold zero-padding and weighted neighborhood centroids; exact MATLAB bin counts, peak weighting, tie behavior, and interpolation must be compared against the original functions before claiming parity.
- Morphological/connected-component details may differ from MATLAB Image Processing Toolbox behavior, especially connectivity conventions and boundary treatment.
- The orientation convention must be confirmed against MATLAB outputs and the intended interpretation of the biological pattern.
- Current synthetic tests establish selected expected frequencies, directions, masks, and merge behavior; they do not prove whole-pipeline parity on real microscopy data.

## Fixture Plan

For each supplied MATLAB fixture, record the source image hash, MATLAB version, toolbox version, `w`, `dL`, `minL`, `maxL`, filter settings, and output hash. Compare window coordinates first, then per-window wavelength/direction/power/uniqueness/range masks, and finally merged fields. Use explicit numeric tolerances for floating-point fields and exact comparisons for coordinates and masks. Do not include images or outputs unless their redistribution rights are clear.
