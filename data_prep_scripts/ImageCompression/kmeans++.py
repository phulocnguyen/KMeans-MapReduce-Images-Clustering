import argparse
from os.path import join
from pathlib import Path

import cv2
import numpy as np

# parser = argparse.ArgumentParser(description='This script creates points.txt and clusters.txt files for a given image.')

# parser.add_argument('--src_img', type=str, help='Path to the source image.')
# parser.add_argument('--dst_folder', type=str, help='Directory in which points.txt and clusters.txt will be saved.')
# parser.add_argument('--k_init_centroids', type=int, help='How many initial centroids to generate using KMeans++.',
#                     default=30)

# args = parser.parse_args()

def nparray_to_str(X):
    to_save = '\n'.join([' '.join(str(X[i])[1:-1].split()) for i in range(len(X))])
    return to_save

def kmeans_plus_plus_init(points, k):
    """
    Initialize centroids using KMeans++ strategy.
    
    :param points: np.array, array of data points (N, D)
    :param k: int, number of centroids to generate
    :return: np.array, initialized centroids (k, D)
    """
    n_points = points.shape[0]
    centroids = []

    first_idx = np.random.randint(0, n_points)
    centroids.append(points[first_idx])

    for _ in range(1, k):
        distances = np.array([min(np.sum((p - c) ** 2) for c in centroids) for p in points])

        probabilities = distances / np.sum(distances)
        next_idx = np.random.choice(n_points, p=probabilities)
        centroids.append(points[next_idx])

    return np.array(centroids)

def main(src_img, dst_folder, k):
    # Files to be created
    points_path = join(dst_folder, 'points.txt')
    clusters_path = join(dst_folder, 'clusters.txt')

    # Create directory
    Path(dst_folder).mkdir(parents=True, exist_ok=True)

    # Load and write points
    img = cv2.imread(src_img).reshape((-1, 3)).astype(np.float32)
    with open(points_path, 'w') as f:
        f.write(nparray_to_str(img))
    print(f'Points saved in: {points_path}')

    # Generate and save centroids using KMeans++
    centroids = kmeans_plus_plus_init(img, k)
    tmp_labels = np.arange(1, k + 1).reshape((k, 1))
    clusters = np.hstack((tmp_labels, centroids))

    with open(clusters_path, 'w') as f:
        f.write(nparray_to_str(clusters))
    print(f'Centroids saved in: {clusters_path}')

# if __name__ == '__main__':
#     args = parser.parse_args()
#     main(data_prep_scripts/ImageCompression/sample_images/img.jpg, resources/input/ImageCompression)

src_img = 'data_prep_scripts/ImageCompression/sample_images/img.jpg'
dst_folder = 'resources/input/ImageCompression/kmeans++'
k = 10
main(src_img, dst_folder, k)