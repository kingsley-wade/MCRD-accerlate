"""LPP demo: analyze a synthetic image with known wavelength/orientation."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from mcrd_accel.config import LPPConfig
from mcrd_accel.lpp import analyze


def create_synthetic_image(
    size: int = 256,
    wavelength: float = 16.0,
    angle_deg: float = 45.0,
    noise: float = 0.1,
) -> np.ndarray:
    """Create a synthetic sinusoid image with known wavelength and orientation."""
    x = np.arange(size)
    y = np.arange(size)
    X, Y = np.meshgrid(x, y)

    # Rotate coordinates
    angle_rad = np.deg2rad(angle_deg)
    X_rot = X * np.cos(angle_rad) + Y * np.sin(angle_rad)

    # Sinusoid
    frequency = 1.0 / wavelength
    image = np.sin(2 * np.pi * frequency * X_rot).astype(np.float32)

    # Add noise
    if noise > 0:
        rng = np.random.default_rng(42)
        image += rng.normal(0, noise, image.shape).astype(np.float32)

    return image


def main():
    print("LPP Analysis Demo: Synthetic Image")
    print("=" * 60)

    # Create synthetic image
    wavelength_true = 16.0  # pixels
    angle_true = 45.0  # degrees
    image = create_synthetic_image(size=256, wavelength=wavelength_true, angle_deg=angle_true, noise=0.1)

    print(f"Image size: {image.shape}")
    print(f"True wavelength: {wavelength_true} pixels")
    print(f"True orientation: {angle_true}°")
    print()

    # Configure LPP
    config = LPPConfig(
        w=32,  # window size
        dL=8,  # stride
        minL=8.0,  # minimum wavelength
        maxL=32.0,  # maximum wavelength
        batch_size=32,
        peak_ratio=0.5,
    )

    print(f"LPP config: w={config.w}, dL={config.dL}, minL={config.minL}, maxL={config.maxL}")
    print()

    # Run analysis
    print("Running LPP analysis...")
    result = analyze(image, config)

    print(f"Valid pixels: {np.sum(result.validity)} / {result.validity.size} ({100*np.sum(result.validity)/result.validity.size:.1f}%)")
    print()

    # Compute statistics on valid pixels
    valid_wl = result.wavelength[result.validity]
    valid_dir = result.direction[result.validity]

    if len(valid_wl) > 0:
        wl_mean = np.mean(valid_wl)
        wl_std = np.std(valid_wl)
        dir_mean = np.mean(valid_dir)
        dir_std = np.std(valid_dir)

        print(f"Measured wavelength: {wl_mean:.2f} ± {wl_std:.2f} pixels (true: {wavelength_true})")
        print(f"Measured orientation: {np.rad2deg(dir_mean):.1f}° ± {np.rad2deg(dir_std):.1f}° (true: {angle_true}°)")
        print()

        # Error
        wl_error = abs(wl_mean - wavelength_true) / wavelength_true * 100
        dir_error = abs(np.rad2deg(dir_mean) - angle_true)
        print(f"Wavelength error: {wl_error:.1f}%")
        print(f"Orientation error: {dir_error:.1f}°")

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Original image
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Synthetic Image")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    # Wavelength field
    im1 = axes[1].imshow(result.wavelength, cmap="viridis", vmin=config.minL, vmax=config.maxL)
    axes[1].set_title("Wavelength Field")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    plt.colorbar(im1, ax=axes[1], label="Wavelength (pixels)")

    # Direction field
    im2 = axes[2].imshow(result.direction, cmap="hsv", vmin=0, vmax=np.pi)
    axes[2].set_title("Direction Field")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    plt.colorbar(im2, ax=axes[2], label="Direction (radians)")

    plt.tight_layout()
    plt.savefig("reports/lpp_demo.png", dpi=150, bbox_inches="tight")
    print(f"\nVisualization saved to reports/lpp_demo.png")


if __name__ == "__main__":
    main()
