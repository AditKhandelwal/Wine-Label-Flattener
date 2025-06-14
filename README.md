# Wine Label Flattener

This project flattens curved wine bottle labels from a single photo using computer vision and geometric interpolation. The output is a straightened label image suitable for OCR.

## Pipeline Overview
1. **Preprocess Image** – Resize and clean input photo.
2. **Label Detection (U-Net)** – Predict mask of the label.
3. **Cylinder Edge Detection** – Fit a cylindrical mesh on the label.
4. **Unwarp** – Flatten the curved label to a 2D image.
5. **OCR** – Extract wine name, vintage, etc.

## Getting Started
```bash
pip install -r requirements.txt
python src/step1_preprocess.py
