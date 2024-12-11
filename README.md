# Images Clustering by Kmeans based on MapReduce Algorithm

## Overview
This repository implements a distributed version of the KMeans algorithm using the MapReduce framework. The application focuses on two primary use cases:

#Collaborators
1. Nguyễn Phú Lộc
2. Phạm Chiến
3. Bùi Ngọc Khánh
4. Nguyễn Quang Huy 

1. **Image Clustering:** Grouping pixels or image features into clusters for segmentation or categorization.
2. **Image Compression:** Reducing image size by representing similar pixel values with their cluster centroids.

By leveraging MapReduce, the process is highly scalable and can efficiently handle large datasets and high-resolution images.


---

## How It Works

### Input Format
- **Images:** Input images should be stored in a distributed file system (e.g., HDFS) in standard formats like `.jpg`, `.png`, etc.
- **Key-Value Representation:** Images are split into smaller data chunks with each chunk represented as key-value pairs:
  - **Key:** Byte offset or image identifier.
  - **Value:** Pixel or feature data in serialized form.

### MapReduce Workflow
1. **Mapper:**
   - Processes each image chunk or pixel batch.
   - Assigns each data point to the nearest cluster centroid.
   - Emits intermediate results as key-value pairs: cluster ID (key) and associated data points (value).

2. **Reducer:**
   - Aggregates intermediate data for each cluster.
   - Computes new centroids based on the aggregated data.
   - Emits updated cluster centroids.

3. **Iteration:**
   - The process iterates until the centroids converge or a specified number of iterations is reached.

### Output
- **Clustered Images:** Images segmented into regions based on cluster assignments.
- **Compressed Images:** Images reconstructed using the cluster centroids, reducing storage size.

---

## Setup and Usage

### Prerequisites
- Python 3.x
- Hadoop or a similar MapReduce framework
- Required libraries: `numpy`, `pillow`, `matplotlib`

### Installation
Clone the repository:
```bash
git clone https://github.com/phulocnguyen/KMeans-MapReduce-Images_Clustering.git
cd KMeans-MapReduce-Images_Clustering
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### Configuration
Set parameters in the `run.sh` file:
- JAR_PATH: path of built jar
- MAIN_CLASS=Main
- INPUT_FILE_PATH: points file path
- STATE_PATH: cluster file path
- NUMBER_OF_REDUCERS=3
- OUTPUT_DIR: path of output
- DELTA: threshold to stop the iteration
- MAX_ITERATIONS: maximum number of iterations
- DISTANCE: type of distance metrics

### Running the Code
1. **Preprocess Images:**
   Convert images to pixel or feature representation for processing.
   ```bash
   python data_prep_script/.../data_prep_compress.py --src_img /path/to/images --dst_folder /path/to/output --k_init_centriods 10
   ```

2. **Run MapReduce:**
   Execute the KMeans algorithm with MapReduce.
   ```bash
   bash run.sh
   ```

---

## Examples

### Input Image
![Input Image](data_prep_scripts/ImageCompression/sample_images/img.jpg)

### Compressed Image
![Compressed Image](result.jpg)

### Clustering Images
![Clustering Image](result.jpg)



---

## Limitations
- High network overhead for large datasets in non-optimized environments.
- Initial centroid selection can affect the final clustering results.

---

## Future Work
- Optimize data serialization for faster MapReduce operations.
- Extend to hierarchical clustering for multi-level image segmentation.
- Support for additional image formats and feature extraction techniques.

---