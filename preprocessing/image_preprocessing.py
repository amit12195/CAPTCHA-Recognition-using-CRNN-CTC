import cv2
import os
import numpy as np

def preprocess_image(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"Skipped: {filename}")
            continue

    
        img = cv2.medianBlur(img, 3)

        ##---- Adaptive Thresholding to highlight digits
        thresh = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )


        kernel = np.ones((2, 2), np.uint8)
        clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        cv2.imwrite(output_path, clean)

    print(" Preprocessing done.")


preprocess_image("test_data", "data_12Apr/processed")