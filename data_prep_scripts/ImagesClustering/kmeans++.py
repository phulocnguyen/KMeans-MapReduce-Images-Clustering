import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from sklearn.preprocessing import StandardScaler

# Hàm tải ảnh và trích xuất đặc trưng
def load_images_and_extract_features(image_folder, image_size=(224, 224)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Chuẩn bị mô hình ResNet-50
    model = resnet50(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])  # Bỏ lớp cuối (FC layer)
    model.eval().to(device)
    
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    features = []
    
    for file_name in os.listdir(image_folder):
        if file_name.endswith(('.jpg', '.png', '.jpeg')):
            file_path = os.path.join(image_folder, file_name)
            
            # Load và xử lý ảnh
            img = transforms.ToPILImage()(torch.randn(3, *image_size))  # debug dummy file
            img = transform(img).unsqueeze(0).to(device)
            
            # Trích xuất đặc trưng
            with torch.no_grad():
                feature = model(img).squeeze().cpu().numpy()
            features.append(feature)
    
    return np.array(features)

# Hàm chuẩn hóa đặc trưng
def scale_features(features):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features, scaler

# Hàm lưu đặc trưng vào file txt
def save_features_to_txt(features, output_path):
    with open(output_path, 'w') as f:
        for feature in features:
            feature_line = " ".join(map(str, feature))
            f.write(feature_line + "\n")

# Hàm tính khoảng cách euclidean
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# Hàm khởi tạo tâm cụm bằng K-means++
def initialize_centroids_kmeans_plus_plus(features, n_clusters):
    n_samples, n_features = features.shape
    centroids = np.zeros((n_clusters, n_features))
    
    # Chọn centroid đầu tiên ngẫu nhiên
    centroids[0] = features[np.random.randint(n_samples)]
    
    # Chọn các centroid còn lại
    for i in range(1, n_clusters):
        # Tính khoảng cách từ mỗi điểm đến centroid gần nhất
        distances = np.array([
            min([euclidean_distance(x, c) for c in centroids[:i]])
            for x in features
        ])
        
        # Tính xác suất chọn mỗi điểm
        probabilities = distances ** 2
        probabilities = probabilities / probabilities.sum()
        
        # Chọn centroid tiếp theo
        centroids[i] = features[np.random.choice(n_samples, p=probabilities)]
    
    return centroids

# Hàm lưu tâm cụm vào file txt kèm clusterid
def save_centroids_to_txt_with_clusterid(centroids, output_path):
    with open(output_path, 'w') as f:
        for cluster_id, centroid in enumerate(centroids):
            centroid_line = f"{cluster_id} " + " ".join(map(str, centroid))
            f.write(centroid_line + "\n")

# Chạy chương trình
if __name__ == "__main__":
    # Đường dẫn tới thư mục chứa ảnh
    image_folder = "data_prep_scripts/ImagesClustering/dataset"  # Thay bằng đường dẫn tới thư mục ảnh
    
    # Trích xuất đặc trưng từ ảnh
    features = load_images_and_extract_features(image_folder)
    
    # Chuẩn hóa đặc trưng
    scaled_features, scaler = scale_features(features)
    
    # Lưu đặc trưng vào file txt
    points_output_path = "resources/input/ImageClustering/points.txt"
    save_features_to_txt(scaled_features, points_output_path)
    print(f"Features saved to {points_output_path}")
    
    # Khởi tạo tâm cụm bằng K-means++
    n_clusters = 5
    centroids = initialize_centroids_kmeans_plus_plus(scaled_features, n_clusters)
    
    # Lưu tâm cụm vào file txt với clusterid
    centroids_output_path = "resources/input/ImageClustering/cluster++.txt"
    save_centroids_to_txt_with_clusterid(centroids, centroids_output_path)
    print(f"Centroids with cluster IDs saved to {centroids_output_path}")