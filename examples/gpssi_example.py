"""Example usage of gpssi for 2D image segmentation sampling.

This example demonstrates how to:
1. Load an image and segmentation
2. Compute geodesic distance map
3. Setup kernel and covariance
4. Generate multiple segmentation samples
"""

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import gpssi

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent


def main() -> None:
    """Run the 2D sampling example."""
    data_dir = SCRIPT_DIR / "data"
    np_img, np_seg = load_data(
        str(data_dir / "example_img.png"), str(data_dir / "example_seg.png")
    )
    np_img, np_seg = np_img[110:180, 100:180], np_seg[110:180, 100:180]

    # Compute geodesic distance map
    # Using RASTER_SCAN method for geodesic distances considering image gradients
    # Alternatives: EUCLIDEAN (fast, ignores image), FAST_MARCHING (hybrid)
    np_geo = gpssi.get_geodesic_map(
        np_img, np_seg, lmbda=0.9, iterations=2, method=gpssi.GeodesicMethod.RASTER_SCAN
    )
    np_geo[np_img == 0] = np_geo.max()  # set background dist to max

    # Kernel parameters
    # w1: length scale - radius of circle with equal area to segmentation
    w1 = np.sqrt((np_seg > 0).sum() / np.pi)  # radius r=(A/pi)^(1/2)
    # w0: amplitude - controls variance of samples
    w0 = 5  # or (D/2)**2, where D is the expected distance in np_geo for the 95% CI
    kernel = gpssi.RbfKernel(w0, w1, eps=1e-8)

    # Create covariance using Kronecker representation (memory efficient)
    cov = gpssi.get_covariance(np_img.shape, kernel, cov_repr="kron")
    # Alternative: full covariance (only for small images)
    # cov = gpssi.get_covariance(np_img.shape, kernel, cov_repr='full')

    # Generate samples
    samples = []
    for _i in range(5):
        noise_vec = np.random.randn(np_geo.size)
        sample = gpssi.get_sample(np_geo, cov, noise_vec)
        samples.append(sample)

    # Plotting
    plot_img(np_geo, "jet", colorbar=True)
    plot_mask_overlay(np_img, np_seg)
    for sample in samples:
        plot_mask_overlay(np_img, sample)


def plot_img(arr: np.ndarray, cmap: str = "gray", *, colorbar: bool = False) -> None:
    """Plot an image array."""
    fig, ax = plt.subplots()
    im = ax.imshow(arr, cmap=cmap)
    ax.set_axis_off()
    if colorbar:
        fig.colorbar(im, ax=ax)
    plt.show()
    plt.close(fig)


def plot_mask_overlay(img: np.ndarray, mask: np.ndarray) -> None:
    """Plot image with mask overlay."""
    fig, ax = plt.subplots()
    overlay(ax, img, mask)
    plt.show()
    plt.close(fig)


def overlay(ax: plt.Axes, img: np.ndarray, mask: np.ndarray) -> None:
    """Overlay mask on image."""
    ax.imshow(img, cmap="gray")
    ma_sample = np.ma.masked_equal(mask, 0)
    ax.imshow(ma_sample, cmap="Reds", alpha=0.5, vmin=0, vmax=1)
    ax.set_axis_off()


def load_data(img_path: str, seg_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load image and segmentation from files."""
    np_img = np.asarray(Image.open(img_path)).astype(np.float32)
    np_seg = np.asarray(Image.open(seg_path)).astype(np.uint8) // 255
    return np_img, np_seg


if __name__ == "__main__":
    main()
