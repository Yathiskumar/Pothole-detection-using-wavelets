import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt

# Step 1: Load the road image
image_path = 'images (2).jpeg'  # Replace with your uploaded image
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
print(image.shape)
# Step 2: Enhance contrast using histogram equalization
image = cv2.equalizeHist(image)

# Step 3: Apply wavelet transform
def apply_wavelet_transform(image, wavelet, level):
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    coeffs[0] = np.zeros_like(coeffs[0])  # Remove approximation coefficients
    reconstructed_image = pywt.waverec2(coeffs, wavelet)
    return np.uint8(np.clip(reconstructed_image, 0, 255))


wavelet_image = apply_wavelet_transform(image, wavelet='db2', level=3)

# Step 4: Identify regions different from the background
mean_background = np.mean(wavelet_image)
std_dev_background = np.std(wavelet_image)
difference_mask = np.abs(wavelet_image - mean_background) > (1.5 * std_dev_background)  # Tuned threshold
difference_mask = np.uint8(difference_mask * 255)  # Convert mask to binary image
difference_mask = cv2.medianBlur(difference_mask, 5)

# Visualize the binary mask
plt.figure(figsize=(10, 5))
plt.title("Binary Mask (Difference Regions)")
plt.imshow(difference_mask, cmap='gray')
plt.show()

# Step 5: Contour detection
contours, _ = cv2.findContours(difference_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

pothole_detected = False
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 50:  # Lower threshold for area
        pothole_detected = True
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

# Step 6: Output result
if pothole_detected:
    print("Pothole Detected")
else:
    print("No Pothole Detected")

# Step 7: Display results
plt.figure(figsize=(15, 10))
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')

plt.subplot(1, 3, 2)
plt.title("Wavelet Transformed Image")
plt.imshow(wavelet_image, cmap='gray')

