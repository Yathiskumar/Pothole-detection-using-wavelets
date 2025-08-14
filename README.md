### Pothole Detection (Wavelet-based)

A simple, headless pothole detection pipeline using wavelet-domain anomaly detection. It loads a road image, enhances contrast, highlights high-frequency anomalies, denoises with a median filter, then detects connected components as candidate potholes.

### Install

- System Python is PEP 668 managed; either use a venv or override. Quickest (already tested here):
```bash
python3 -m pip install --break-system-packages -r requirements.txt
```

### Usage

```bash
python3 Code_project.py \
  --image "images (2).jpeg" \
  --wavelet db2 \
  --level 3 \
  --sigma-k 1.5 \
  --kernel-size 5 \
  --min-area 50 \
  --connectivity 4 \
  --output-dir .
```

- **Outputs**: saves `results.png` and `binary_mask.png` to `--output-dir`.
- **Headless**: uses Matplotlib Agg backend; no GUI windows.

### Flags

- **--image**: path to the input image
- **--wavelet**: wavelet family (e.g., `db2`, `haar`)
- **--level**: wavelet decomposition level
- **--sigma-k**: threshold multiplier on std-dev for anomaly mask
- **--kernel-size**: median filter window size (must be odd)
- **--min-area**: minimum connected component area to keep
- **--connectivity**: 4 or 8 connectivity for components
- **--no-intermediate**: skip saving `binary_mask.png`
- **--output-dir**: where to write images

### Notes

- The median filter and connected components are implemented in pure NumPy for minimal dependencies.
- For large images, you can downscale before processing for speed.
- Tuning `--sigma-k` up reduces sensitivity (fewer detections), down increases.