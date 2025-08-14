import argparse
import os
from typing import List, Tuple, Dict

import numpy as np
import pywt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image


def equalize_histogram(gray_image: np.ndarray) -> np.ndarray:
    if gray_image.dtype != np.uint8:
        raise ValueError("equalize_histogram expects a uint8 image")
    hist = np.bincount(gray_image.flatten(), minlength=256)
    cdf = hist.cumsum()
    nonzero_mask = cdf != 0
    cdf_min = cdf[nonzero_mask][0] if np.any(nonzero_mask) else 0
    num_pixels = gray_image.size
    cdf_scaled = ((cdf - cdf_min) / max(num_pixels - cdf_min, 1) * 255.0)
    cdf_scaled = np.clip(cdf_scaled, 0, 255).astype(np.uint8)
    return cdf_scaled[gray_image]


def apply_wavelet_transform(image: np.ndarray, wavelet: str, level: int) -> np.ndarray:
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    coeffs = list(coeffs)
    coeffs[0] = np.zeros_like(coeffs[0])
    reconstructed_image = pywt.waverec2(coeffs, wavelet)
    return np.uint8(np.clip(reconstructed_image, 0, 255))


def median_filter(binary_image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")
    pad = kernel_size // 2
    padded = np.pad(binary_image, ((pad, pad), (pad, pad)), mode='edge')
    output = np.empty_like(binary_image)
    for i in range(binary_image.shape[0]):
        row_slices = []
        for ki in range(kernel_size):
            row_slices.append(padded[i + ki, :])
        stacked_rows = np.stack(row_slices, axis=0)
        for j in range(binary_image.shape[1]):
            window = stacked_rows[:, j:j + kernel_size].reshape(-1)
            output[i, j] = np.median(window)
    return output.astype(np.uint8)


def find_connected_components(binary_mask: np.ndarray, connectivity: int = 4) -> List[Dict[str, Tuple[int, int, int, int]]]:
    mask = binary_mask > 0
    visited = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    components: List[Dict[str, Tuple[int, int, int, int]]] = []

    if connectivity == 8:
        neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    else:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for y in range(height):
        for x in range(width):
            if mask[y, x] and not visited[y, x]:
                stack = [(y, x)]
                visited[y, x] = True
                area = 0
                min_y, min_x = y, x
                max_y, max_x = y, x
                while stack:
                    cy, cx = stack.pop()
                    area += 1
                    if cy < min_y:
                        min_y = cy
                    if cx < min_x:
                        min_x = cx
                    if cy > max_y:
                        max_y = cy
                    if cx > max_x:
                        max_x = cx
                    for dy, dx in neighbors:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
                components.append({
                    'area': area,
                    'bbox': (min_x, min_y, max_x - min_x + 1, max_y - min_y + 1)
                })
    return components


def process_image(
    image_path: str,
    wavelet: str = 'db2',
    level: int = 3,
    sigma_k: float = 1.5,
    kernel_size: int = 5,
    min_area: int = 50,
    connectivity: int = 4,
    output_dir: str = '.',
    save_intermediate: bool = True
) -> Dict[str, object]:
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image_pil = Image.open(image_path).convert('L')
    image = np.array(image_pil, dtype=np.uint8)
    print(f"Loaded image with shape: {image.shape}")

    image_eq = equalize_histogram(image)

    wavelet_image = apply_wavelet_transform(image_eq, wavelet=wavelet, level=level)

    mean_background = np.mean(wavelet_image)
    std_dev_background = np.std(wavelet_image)
    difference_mask = np.abs(wavelet_image - mean_background) > (sigma_k * std_dev_background)
    difference_mask = (difference_mask.astype(np.uint8) * 255)

    difference_mask = median_filter(difference_mask, kernel_size=kernel_size)

    os.makedirs(output_dir, exist_ok=True)

    binary_mask_path = os.path.join(output_dir, 'binary_mask.png')
    if save_intermediate:
        plt.figure(figsize=(10, 5))
        plt.title("Binary Mask (Difference Regions)")
        plt.imshow(difference_mask, cmap='gray')
        plt.tight_layout()
        plt.savefig(binary_mask_path)
        plt.close()

    components = find_connected_components(difference_mask, connectivity=connectivity)

    pothole_detected = False
    rectangles: List[Tuple[int, int, int, int]] = []
    for comp in components:
        area = comp['area']
        if area > min_area:
            pothole_detected = True
            rectangles.append(comp['bbox'])

    results_path = os.path.join(output_dir, 'results.png')
    plt.figure(figsize=(15, 10))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title("Wavelet Transformed Image")
    plt.imshow(wavelet_image, cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title("Detections")
    plt.imshow(image, cmap='gray')
    ax = plt.gca()
    for (x, y, w, h) in rectangles:
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor='lime', linewidth=2))
    plt.tight_layout()
    plt.savefig(results_path)
    plt.close()

    print("Pothole Detected" if pothole_detected else "No Pothole Detected")
    saved = [results_path]
    if save_intermediate:
        saved.append(binary_mask_path)
    print("Saved:", ", ".join(saved))

    return {
        'pothole_detected': pothole_detected,
        'rectangles': rectangles,
        'results_path': results_path,
        'binary_mask_path': binary_mask_path if save_intermediate else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Pothole detection using wavelet-based anomaly detection')
    parser.add_argument('--image', '-i', type=str, default='images (2).jpeg', help='Path to input road image')
    parser.add_argument('--wavelet', type=str, default='db2', help='Wavelet name (e.g., db2, haar)')
    parser.add_argument('--level', type=int, default=3, help='Wavelet decomposition level')
    parser.add_argument('--sigma-k', type=float, default=1.5, help='Threshold multiplier on background std-dev')
    parser.add_argument('--kernel-size', type=int, default=5, help='Median filter kernel size (odd)')
    parser.add_argument('--min-area', type=int, default=50, help='Minimum connected component area to keep')
    parser.add_argument('--connectivity', type=int, choices=[4, 8], default=4, help='Connectivity for components')
    parser.add_argument('--output-dir', type=str, default='.', help='Directory to save output images')
    parser.add_argument('--no-intermediate', action='store_true', help='Do not save intermediate binary mask image')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_image(
        image_path=args.image,
        wavelet=args.wavelet,
        level=args.level,
        sigma_k=args.sigma_k,
        kernel_size=args.kernel_size,
        min_area=args.min_area,
        connectivity=args.connectivity,
        output_dir=args.output_dir,
        save_intermediate=not args.no_intermediate,
    )


if __name__ == '__main__':
    main()

