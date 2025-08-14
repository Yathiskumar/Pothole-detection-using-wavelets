import numpy as np
import pywt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

# Step 1: Load the road image
image_path = 'images (2).jpeg'  # Replace with your uploaded image
image_pil = Image.open(image_path).convert('L')
image = np.array(image_pil, dtype=np.uint8)
print(f"Loaded image with shape: {image.shape}")

# Step 2: Enhance contrast using histogram equalization (NumPy implementation)
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

image = equalize_histogram(image)

# Step 3: Apply wavelet transform

def apply_wavelet_transform(image: np.ndarray, wavelet: str, level: int) -> np.ndarray:
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    coeffs = list(coeffs)
    coeffs[0] = np.zeros_like(coeffs[0])  # Remove approximation coefficients
    reconstructed_image = pywt.waverec2(coeffs, wavelet)
    return np.uint8(np.clip(reconstructed_image, 0, 255))

wavelet_image = apply_wavelet_transform(image, wavelet='db2', level=3)

# Step 4: Identify regions different from the background
mean_background = np.mean(wavelet_image)
std_dev_background = np.std(wavelet_image)
difference_mask = np.abs(wavelet_image - mean_background) > (1.5 * std_dev_background)  # Tuned threshold
difference_mask = (difference_mask.astype(np.uint8) * 255)  # Convert mask to binary image [0,255]

# Median filter implementation (5x5) without SciPy/OpenCV

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

difference_mask = median_filter(difference_mask, kernel_size=5)

# Visualize the binary mask
plt.figure(figsize=(10, 5))
plt.title("Binary Mask (Difference Regions)")
plt.imshow(difference_mask, cmap='gray')
plt.tight_layout()
plt.savefig("binary_mask.png")
plt.close()

# Step 5: Contour/Component detection (connected components)

def find_connected_components(binary_mask: np.ndarray, connectivity: int = 4):
    mask = binary_mask > 0
    visited = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    components = []

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
                    if cy < min_y: min_y = cy
                    if cx < min_x: min_x = cx
                    if cy > max_y: max_y = cy
                    if cx > max_x: max_x = cx
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

components = find_connected_components(difference_mask, connectivity=4)

pothole_detected = False
rectangles = []
for comp in components:
    area = comp['area']
    if area > 50:  # Lower threshold for area
        pothole_detected = True
        rectangles.append(comp['bbox'])  # (x, y, w, h)

# Step 6: Output result
if pothole_detected:
    print("Pothole Detected")
else:
    print("No Pothole Detected")

# Step 7: Display and save results
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
plt.savefig("results.png")
plt.close()
print("Saved visualizations: binary_mask.png, results.png")

