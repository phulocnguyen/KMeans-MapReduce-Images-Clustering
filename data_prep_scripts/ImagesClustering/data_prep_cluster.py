import argparse
import os
from os.path import join
from pathlib import Path

import cv2
from PIL import Image
import numpy as np
from imgbeddings import imgbeddings

parser = argparse.ArgumentParser(description='This script creates features.txt and clusters.txt files for image clustering.')

parser.add_argument('--src_folder', type=str, help='Path to the folder containing images.')
parser.add_argument('--dst_folder', type=str, help='Directory in which features.txt and clusters.txt will be saved.')
parser.add_argument('--k_init_centroids', type=int, help='How many initial uniformly sampled centroids to generate.',
                    default=10)

args = parser.parse_args()


def nparray_to_str(X):
    """Convert a numpy array to a space-separated string."""
    to_save = '\n'.join([' '.join(str(X[i])[1:-1].split()) for i in range(len(X))])
    return to_save


def main(src_folder, dst_folder, k):
    # files to be created
    features_path = join(dst_folder, 'features.txt')
    clusters_path = join(dst_folder, 'clusters.txt')

    # create directory
    Path(dst_folder).mkdir(parents=True, exist_ok=True)

    ibed = imgbeddings()

    # Process all images in the folder
    all_features = []
    for filename in os.listdir(src_folder):
        img_path = join(src_folder, filename)
        img = cv2.imread(img_path)
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        print(f"Processing image: {img_path}")
        embedding = ibed.to_embeddings(img_pil).squeeze() 
        all_features.append(embedding)

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
