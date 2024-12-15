import os
import shutil
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from PIL import Image

def load_centroids(centroids_file):
    """Đọc tọa độ các tâm cluster từ file"""
    centroids = []
    cluster_ids = []
    with open(centroids_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 1:
                cluster_id = int(parts[0])
                centroid = np.array([float(x) for x in parts[1:]])
                centroids.append(centroid)
                cluster_ids.append(cluster_id)
    return np.array(centroids), cluster_ids

def extract_features(image_path, model, transform, device):
    """Trích xuất đặc trưng từ một ảnh"""
    img = Image.open(image_path)
    img = img.convert('RGB')
    img = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        feature = model(img).squeeze().cpu().numpy()
    return feature

def setup_feature_extractor(device):
    """Chuẩn bị model và transform để trích xuất đặc trưng"""
    model = resnet50(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval().to(device)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225]),
    ])
    
    return model, transform

def assign_clusters(features, centroids):
    """Gán mỗi ảnh vào cluster gần nhất"""
    distances = np.linalg.norm(features[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)

def organize_and_visualize_clusters(image_folder, assignments, cluster_ids, output_folder, 
                                  max_images_per_cluster=5):
    """Tổ chức và hiển thị ảnh theo cluster"""
    # Tạo thư mục output
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)
    
    # Tạo thư mục cho từng cluster và copy ảnh
    image_files = [f for f in os.listdir(image_folder) 
                  if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    cluster_paths = {}
    for cluster_id in np.unique(assignments):
        cluster_name = f'cluster_{cluster_ids[cluster_id]}'
        cluster_path = os.path.join(output_folder, cluster_name)
        os.makedirs(cluster_path)
        cluster_paths[cluster_id] = cluster_path
    
    # Copy ảnh vào các cluster
    for img_idx, (img_file, cluster_id) in enumerate(zip(image_files, assignments)):
        src_path = os.path.join(image_folder, img_file)
        dst_path = os.path.join(cluster_paths[cluster_id], img_file)
        shutil.copy2(src_path, dst_path)
    
    # Visualize clusters
    cluster_folders = sorted(os.listdir(output_folder))
    n_clusters = len(cluster_folders)
    
    fig = plt.figure(figsize=(15, 3*n_clusters))
    
    for idx, cluster_folder in enumerate(cluster_folders):
        cluster_path = os.path.join(output_folder, cluster_folder)
        images = [f for f in os.listdir(cluster_path) 
                 if f.lower().endswith(('.jpg', '.png', '.jpeg'))][:max_images_per_cluster]
        
        for img_idx, img_file in enumerate(images):
            img_path = os.path.join(cluster_path, img_file)
            img = Image.open(img_path)
            
            plt.subplot(n_clusters, max_images_per_cluster, 
                       idx * max_images_per_cluster + img_idx + 1)
            plt.imshow(img)
            plt.axis('off')
            if img_idx == 0:
                n_images = len([f for f in os.listdir(cluster_path) 
                              if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
                plt.title(f'{cluster_folder}\n({n_images} images)', pad=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'cluster_visualization.png'))
    plt.show()
    
    # In thống kê
    print("\nThống kê cluster:")
    print("-" * 30)
    total_images = 0
    for cluster_folder in cluster_folders:
        cluster_path = os.path.join(output_folder, cluster_folder)
        n_images = len([f for f in os.listdir(cluster_path) 
                       if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        total_images += n_images
        print(f"{cluster_folder}: {n_images} ảnh")
    print("-" * 30)
    print(f"Tổng số: {total_images} ảnh")

def main():
    # Cấu hình
    image_folder = "/Users/phulocnguyen/Documents/Workspace/KMeans-MapReduce-Images-Clustering/data_prep_scripts/ImagesClustering/dataset"    # Thư mục chứa ảnh
    centroids_file = "/Users/phulocnguyen/Documents/Workspace/KMeans-MapReduce-Images-Clustering/resources/output/Clustering/result.txt"              # File chứa tọa độ centroids
    output_folder = "/Users/phulocnguyen/Documents/Workspace/KMeans-MapReduce-Images-Clustering/resources/output/Clustering/clustered_images"            # Thư mục output
    
    # Thiết lập device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Đọc centroids
    centroids, cluster_ids = load_centroids(centroids_file)
    
    # Chuẩn bị feature extractor
    model, transform = setup_feature_extractor(device)
    
    # Trích xuất đặc trưng từ tất cả ảnh
    print("Extracting features...")
    features = []
    image_files = [f for f in os.listdir(image_folder) 
                  if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)
        feature = extract_features(img_path, model, transform, device)
        features.append(feature)
    
    features = np.array(features)
    
    # Chuẩn hóa features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Chuẩn hóa centroids cùng scale với features
    scaled_centroids = scaler.transform(centroids)
    
    # Gán cluster cho mỗi ảnh
    print("Assigning Cluster to Images...")
    assignments = assign_clusters(scaled_features, scaled_centroids)
    
    # Tổ chức và hiển thị kết quả
    print("Đang tổ chức và hiển thị kết quả...")
    organize_and_visualize_clusters(image_folder, assignments, cluster_ids, output_folder)

if __name__ == "__main__":
    main()