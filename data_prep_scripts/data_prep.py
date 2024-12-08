import argparse
import os
from os.path import join
from pathlib import Path

import cv2
import numpy as np
from skimage.feature import hog

parser = argparse.ArgumentParser(description='This script creates features.txt and clusters.txt files for image clustering.')

parser.add_argument('--src_folder', type=str, help='Path to the folder containing images.')
parser.add_argument('--dst_folder', type=str, help='Directory in which features.txt and clusters.txt will be saved.')
parser.add_argument('--k_init_centroids', type=int, help='How many initial uniformly sampled centroids to generate.',
                    default=10)

args = parser.parse_args()


def nparray_to_str(X):
    """Convert a numpy array to a space-separated string."""
    return '\n'.join([' '.join(map(str, row)) for row in X])

def extract_image_features(img):
    """
    Extracts features from the image.

    :param img: Input image.
    :return: A 1D array representing the image features.
    """
    # Resize image to 64x64
    resized_img = cv2.resize(img, (64, 64))

    # Convert to grayscale for HOG
    gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    # Extract HOG features
    features, _ = hog(
        gray_img, 
        orientations=9, 
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2), 
        block_norm='L2-Hys', 
        visualize=True, 
        feature_vector=True
    )

    return features


def main(src_folder, dst_folder, k):
    # files to be created
    features_path = join(dst_folder, 'points.txt')
    clusters_path = join(dst_folder, 'clusters.txt')

    # create directory
    Path(dst_folder).mkdir(parents=True, exist_ok=True)

    # Process all images in the folder
    all_features = []
    for filename in os.listdir(src_folder):
        img_path = join(src_folder, filename)
        img = cv2.imread(img_path)

        print(f"Processing image: {img_path}")
        features = extract_image_features(img)
        all_features.append(features)

    # write feature points
    all_features = np.array(all_features)
    with open(features_path, 'w') as f:
        f.write(nparray_to_str(all_features))
    print(f'Features saved in: {features_path}')

    # Generate and save uniformly sampled centroids
    s = np.random.uniform(low=all_features.min(), high=all_features.max(), size=(k, all_features.shape[1]))
    tmp_labels = np.arange(1, k + 1).reshape((k, 1))
    clusters = np.hstack((tmp_labels, s))

    with open(clusters_path, 'w') as f:
        f.write(nparray_to_str(clusters))
    print(f'Centroids saved in: {clusters_path}')


if __name__ == '__main__':
    args = parser.parse_args()
    main(args.src_folder, args.dst_folder, args.k_init_centroids)
