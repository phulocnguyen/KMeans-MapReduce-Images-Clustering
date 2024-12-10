import argparse
from glob import glob
from os.path import join, isdir, isfile
import os
from pathlib import Path
import numpy as np
import shutil

parser = argparse.ArgumentParser(description='Cluster images and organize them into folders based on clustering results.')

parser.add_argument('--src_folder', type=str, help='Path to the folder containing images.')
parser.add_argument('--dst_folder', type=str, help='Directory in which features.txt and clusters.txt will be saved.')
parser.add_argument('--dst_img_folder', type=str, help='Path to the image to be written.')

args = parser.parse_args()

def load_clusters(path):
    if isdir(path):
        files = glob(join(path, 'part-r-*[0-9]'))
    elif isfile(path):
        files = [path]
    else:
        raise Exception('Invalid file path.')

    centroids = [load_nparray(file)[:, 1:] for file in files]
    centroids = np.concatenate(centroids, axis=0).reshape(-1, centroids[0].shape[-1])
    return centroids


def load_nparray(file):
    data = []
    with open(file) as f:
        for line in f:
            data.append(np.array([float(num) for num in line.split(' ')]))

    return np.stack(data).astype(np.float64)

def organize_images_by_clusters(image_clustering, src_folder, dst_img_folder):
    files = os.listdir(src_folder)
    """Organize images into subfolders based on cluster labels."""
    for id in range(max(image_clustering)+1):
        cluster_folder = join(dst_img_folder, f"cluster_{id}")
        Path(cluster_folder).mkdir(parents=True, exist_ok=True)
    
    for i, file in enumerate(files):
        src_path = join(src_folder, file)
        cluster_folder = join(dst_img_folder, f"cluster_{image_clustering[i]}")
        shutil.copy(src_path, cluster_folder)

def main(src_folder, dst_folder, dst_img_folder):
    features_path = join(dst_folder, 'points.txt')
    clusters_path = join(dst_folder, 'clusters.txt')

    clusters = load_clusters(clusters_path)
    features = load_nparray(features_path)

    Path(dst_img_folder).mkdir(parents=True, exist_ok=True)

    image_clustering = []
    for i, feature in enumerate(features):
        ind = np.linalg.norm(clusters - feature, axis=-1).argmin()
        image_clustering.append(ind)

    # Organize images into folders
    organize_images_by_clusters(image_clustering, src_folder, dst_img_folder)
    print(f"Images have been organized into folders in: {dst_img_folder}")

if __name__ == '__main__':
    args = parser.parse_args()
    main(args.src_folder, args.dst_folder, args.dst_img_folder)
