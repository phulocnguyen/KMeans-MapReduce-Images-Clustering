#!/usr/bin/env bash

# create directories on hdfs
hadoop fs -mkdir -p /KMeans/resources/input/compression
hadoop fs -mkdir -p /KMeans/resources/output/compression

# copy local input files
hadoop fs -put ./resources/input/ImageCompression/points.txt ./resources/input/ImageCompression/clusters.txt /KMeans/resources/input/compression

# specify input parameters
JAR_PATH=./executable_jar/kmeans_mapreduce_v2.jar
MAIN_CLASS=Main
INPUT_FILE_PATH=/KMeans/resources/input/compression/points.txt
STATE_PATH=/KMeans/resources/input/compression/clusters.txt
NUMBER_OF_REDUCERS=3
OUTPUT_DIR=/KMeans/resources/output/compression 
DELTA=50.0
MAX_ITERATIONS=10
DISTANCE=cosine

hadoop jar ${JAR_PATH} ${MAIN_CLASS} --input ${INPUT_FILE_PATH} \
--state ${STATE_PATH} \
--number ${NUMBER_OF_REDUCERS} \
--output ${OUTPUT_DIR} \
--delta ${DELTA} \
--max ${MAX_ITERATIONS} \
--distance ${DISTANCE}

# execute jar file
LAST_DIR="$(hadoop fs -ls -t -C /KMeans/resources/output/compression | head -1)"

# print results
hadoop fs -cat "$LAST_DIR/part-r-[0-9][0-9][0-9][0-9][0-9]" | sort --numeric --key 1

hadoop fs -ls /KMeans/resources/output/compression/

hadoop fs -get /KMeans/resources/output/compression/ ./resources/output/Compression

python data_prep_scripts/ImageCompression/data_prep_compress.py --src_img data_prep_scripts/ImageCompression/sample_images/img.jpg --dst_folder  resources/input/ImageCompression 

python data_prep_scripts/ImageCompression/visualize_results.py --clusters_path /Users/phulocnguyen/Documents/Workspace/KMeans-MapReduce-Images-Clustering/resources/output/Compression/result.txt --src_img /Users/phulocnguyen/Documents/Workspace/KMeans-MapReduce-Images-Clustering/data_prep_scripts/ImageCompression/sample_images/img.jpg --dst_img /Users/phulocnguyen/Documents/Workspace/KMeans-MapReduce-Images-Clustering/data_prep_scripts/ImageCompression/sample_images/result.jpg