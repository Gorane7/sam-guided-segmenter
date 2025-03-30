import os
import cv2
import numpy as np
import argparse

def merge_masks(folder, start_idx, end_idx):
    merged = None
    # Iterate over indices (inclusive range)
    for i in range(start_idx, end_idx + 1):
        filename = os.path.join(folder, f"{i:05d}.png")
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} not found.")

        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Unable to load {filename}, skipping.")
            continue

        if merged is None:
            # Initialize merged image using the dimensions of the first valid image.
            merged = img.copy()
        else:
            # Merge current mask with the accumulated merged mask.
            merged = np.maximum(merged, img)

    if merged is not None:
        # Save the merged image with a name based on start_idx.
        merged_filename = os.path.join(folder, f"{start_idx:05d}_mask.png")
        cv2.imwrite(merged_filename, merged)
        print(f"Merged image saved as {merged_filename}")
    else:
        print("No valid images found to merge.")
        return

    # Delete the original mask files
    for i in range(start_idx, end_idx + 1):
        filename = os.path.join(folder, f"{i:05d}.png")
        if os.path.exists(filename):
            os.remove(filename)
            print(f"Deleted {filename}")

    # rename the merged file to the original name
    original_filename = os.path.join(folder, f"{start_idx:05d}.png")

if __name__ == "__main__":
    print("Current working directory:", os.getcwd())
    parser = argparse.ArgumentParser(description="Merge image masks into a single mask.")
    parser.add_argument("folder", type=str, help="Folder containing the mask images.")
    parser.add_argument("start_idx", type=int, help="Start index of the mask images.")
    parser.add_argument("end_idx", type=int, help="End index of the mask images.")
    args = parser.parse_args()

    merge_masks(args.folder, args.start_idx, args.end_idx)