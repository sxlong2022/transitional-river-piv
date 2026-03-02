"""
Quick test for C&G repository submission.
Verifies that the core dependencies are installed and the fundamental PIV functions work
using a synthetic displacement example without requiring real data download.
"""

import numpy as np
from openpiv import pyprocess, validation, filters
from src.postprocessing.postprocess import compute_vector_stats

def main():
    print("=== Starting Quick Test ===")
    
    # 1. Generate synthetic images
    # Create a 256x256 image with some random "particles"
    print("1. Generating synthetic image pair...")
    np.random.seed(42)
    frame_a = np.random.randint(0, 255, size=(256, 256), dtype=np.int32)
    
    # Create frame_b by shifting frame_a by (dx=5, dy=-3) pixels
    frame_b = np.roll(frame_a, shift=5, axis=1)  # dx = +5
    frame_b = np.roll(frame_b, shift=-3, axis=0) # dy = -3
    
    # 2. Run OpenPIV extended search area
    print("2. Running OpenPIV cross-correlation...")
    u, v, sig2noise = pyprocess.extended_search_area_piv(
        frame_a,
        frame_b,
        window_size=32,
        overlap=16,
        dt=1.0,
        search_area_size=32,
        sig2noise_method="peak2peak",
    )
    
    x, y = pyprocess.get_coordinates(
        image_size=frame_a.shape,
        search_area_size=32,
        overlap=16,
    )
    
    print(f"   Generated vector field shape: {u.shape}")
    
    # 3. Validation and Filtering
    print("3. Validating and filtering vectors...")
    u, v, mask = validation.sig2noise_val(u, v, sig2noise, threshold=1.3)
    u, v = filters.replace_outliers(u, v, method="localmean")
    
    # Expected displacement is (5, -3) because coordinate system for image is (row, col)
    # OpenPIV returns u (horizontal, cols), v (vertical, rows, reversed based on coords)
    median_u = np.nanmedian(u)
    median_v = np.nanmedian(v)
    print(f"   Median displacement detected: (u={median_u:.2f}, v={median_v:.2f}) pixels")
    
    # 4. Test vector stats computation (mocking a time series of 3 identical pairs)
    print("4. Testing vector statistics aggregation...")
    u_stack = np.stack([u, u, u], axis=0)
    v_stack = np.stack([v, v, v], axis=0)
    
    u_mean, v_mean, u_var, v_var, n_samples, bad_mask = compute_vector_stats(
        u_stack, v_stack, min_samples=2, max_cv=2.0
    )
    
    print(f"   Aggregated mean shape: {u_mean.shape}")
    print(f"   Valid pixels found: {np.sum(~bad_mask)}")
    
    print("=== Quick Test Completed Successfully ==!")
    print("If you see this message, your environment is correctly configured to run the PIV workflow.")

if __name__ == "__main__":
    main()
