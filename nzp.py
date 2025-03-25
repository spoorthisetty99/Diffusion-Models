import numpy as np
import os
import cv2
import argparse

def load_images_from_folder(folder, target_size=(128, 128)):
    images = []
    labels = []
    class_names = sorted(os.listdir(folder))  # Assuming folders are named by class

    for label, class_name in enumerate(class_names):
        class_path = os.path.join(folder, class_name)
        if os.path.isdir(class_path):  # Ensure it's a folder
            for filename in os.listdir(class_path):
                img_path = os.path.join(class_path, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, target_size)  # Resize to a fixed size
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
                    images.append(img)
                    labels.append(label)

    return np.array(images), np.array(labels)

def calculate_statistics(npz_file, save_file=None):
    # Load the dataset
    data = np.load(npz_file)
    stats = {}
    
    for key in data:
        array = data[key]
        if isinstance(array, np.ndarray):
            stats[key] = {
                "mean": np.mean(array, axis=0),
                "std": np.std(array, axis=0),
                "min": np.min(array, axis=0),
                "max": np.max(array, axis=0),
                "median": np.median(array, axis=0)
            }
    
    # Save statistics if required
    if save_file:
        np.savez(save_file, **stats)
        print(f"Statistics saved to {save_file}")
    
    return stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute statistics for a dataset in an NPZ file.")
    parser.add_argument("dataset_path", type=str, help="Path to the image dataset folder.")
    parser.add_argument("npz_file", type=str, help="Path to the output NPZ file.")
    parser.add_argument("--save_stats", type=str, help="Optional path to save computed statistics.")
    
    args = parser.parse_args()
    
    # Load images and check if any were found
    X, y = load_images_from_folder(args.dataset_path)
    if len(X) == 0 or len(y) == 0:
        print("Error: No images found in the dataset path. Check the folder structure.")
        exit(1)
    
    # Save images and labels to NPZ file
    np.savez(args.npz_file, images=X, labels=y)
    print(f"Image dataset saved to {args.npz_file}")
    
    # Compute statistics
    stats = calculate_statistics(args.npz_file, args.save_stats)
    
    for key, stat in stats.items():
        print(f"Statistics for {key}:")
        for metric, value in stat.items():
            print(f"  {metric}: {value}")
        print()
