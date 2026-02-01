"""Example usage of gpssi for 3D image segmentation sampling.

This example demonstrates 3D sampling and requires SimpleITK for loading
NIfTI files. For basic usage without SimpleITK, see gpssi_example.py.

Note: SimpleITK is not included in the default dependencies.
Install with: pip install SimpleITK
"""

import matplotlib.pyplot as plt
import numpy as np

try:
    import SimpleITK as sitk
except ImportError:
    print("SimpleITK is required for this example. Install with: pip install SimpleITK")
    raise

import gpssi


def main() -> None:
    """Run the 3D sampling example."""
    np_img, np_seg, spacing = load_data(
        "data/example_img.nii.gz", "data/example_seg.nii.gz"
    )
    np_img = np_img / np_img.max() * 255  # normalize to be consistent with 2d example
    np_img, np_seg = np_img[0:144, 42:220, 47:192], np_seg[0:144, 42:220, 47:192]

    # Compute geodesic distance map
    # Using RASTER_SCAN method for geodesic distances considering image gradients
    np_geo = gpssi.get_geodesic_map(
        np_img,
        np_seg,
        lmbda=0.9,
        iterations=4,
        spacing=spacing,
        method=gpssi.GeodesicMethod.RASTER_SCAN,
    )
    np_geo[np_img == 0] = np_geo.max()  # set background distance to max

    # Kernel parameters
    # w1: length scale - radius of sphere with equal volume to segmentation
    w1 = ((3 * (np_seg > 0).sum()) / (4 * np.pi)) ** (
        1 / 3
    )  # radius r=(3/4*V/pi)^(1/3)
    # w0: amplitude - controls variance of samples
    w0 = 3  # or (D/2)**2, where D is the expected distance in np_geo for the 95% CI
    kernel = gpssi.RbfKernel(w0, w1, eps=1e-8)

    # Create covariance using Kronecker representation (memory efficient for 3D)
    cov = gpssi.get_covariance(np_img.shape, kernel, cov_repr="kron")

    # Generate samples
    samples = []
    for i in range(5):
        noise_vec = np.random.randn(np_geo.size)
        sample = gpssi.get_sample(np_geo, cov, noise_vec)
        samples.append(sample)

    # Plot slice with largest ground truth area
    sums = np_seg.sum(axis=(1, 2))
    slice_idx = np.argmax(sums)
    plot_img(np_geo[slice_idx], "jet", colorbar=True)
    plot_mask_overlay(np_img[slice_idx], np_seg[slice_idx])
    for sample in samples:
        plot_mask_overlay(np_img[slice_idx], sample[slice_idx])


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


def load_data(
    img_path: str, seg_path: str
) -> tuple[np.ndarray, np.ndarray, tuple[float, ...]]:
    """Load image and segmentation from NIfTI files."""
    img = sitk.ReadImage(img_path)

    np_img = sitk.GetArrayFromImage(img).astype(np.float32)
    # Invert spacing since x and z dims are swapped from sitk to numpy
    spacing = img.GetSpacing()[::-1]

    np_seg = sitk.GetArrayFromImage(sitk.ReadImage(seg_path)).astype(np.uint8)
    return np_img, np_seg, spacing


if __name__ == "__main__":
    main()
