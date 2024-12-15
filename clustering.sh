#!/usr/bin/env bash

# create directories on hdfs
hadoop fs -mkdir -p /KMeans/resources/input/cluster
hadoop fs -mkdir -p /KMeans/resources/output/cluster


#Normal KMeans

# copy local input files
hadoop fs -put ./resources/input/ImageClustering/points.txt ./resources/input/ImageClustering/clusters.txt /KMeans/resources/input/cluster


# specify input parameters
JAR_PATH=./executable_jar/kmeans_mapreduce_v2.jar
MAIN_CLASS=Main
INPUT_FILE_PATH=/KMeans/resources/input/cluster/points.txt
STATE_PATH=/KMeans/resources/input/cluster/clusters.txt
NUMBER_OF_REDUCERS=3
OUTPUT_DIR=/KMeans/resources/output/cluster
DELTA=0.1
MAX_ITERATIONS=10
DISTANCE=euclidean

#KMeans ++

# copy local input files
hadoop fs -put ./resources/input/ImageClustering/points.txt ./resources/input/ImageClustering/cluster++.txt /KMeans/resources/input/cluster


# specify input parameters
JAR_PATH=./executable_jar/kmeans_mapreduce_v2.jar
MAIN_CLASS=Main
INPUT_FILE_PATH=/KMeans/resources/input/cluster/points.txt
STATE_PATH=/KMeans/resources/input/cluster/cluster++.txt
NUMBER_OF_REDUCERS=3
OUTPUT_DIR=/KMeans/resources/output/cluster
DELTA=0.1
MAX_ITERATIONS=10
DISTANCE=euclidean

hadoop jar ${JAR_PATH} ${MAIN_CLASS} --input ${INPUT_FILE_PATH} \
--state ${STATE_PATH} \
--number ${NUMBER_OF_REDUCERS} \
--output ${OUTPUT_DIR} \
--delta ${DELTA} \
--max ${MAX_ITERATIONS} \
--distance ${DISTANCE}

# execute jar file
LAST_DIR="$(hadoop fs -ls -t -C /KMeans/resources/output/cluster | head -1)"

# print results
hadoop fs -cat "$LAST_DIR/part-r-[0-9][0-9][0-9][0-9][0-9]" | sort --numeric --key 1


hadoop fs -get /KMeans/resources/output/cluster ./resources/output/Clustering

python data_prep_scripts/ImageCompression/visualize_results.py 
--clusters_path /Users/phulocnguyen/Documents/Workspace/KMeans-MapReduce-Images-Clustering/resources/output/Compression/result.txt 
--src_img /Users/phulocnguyen/Documents/Workspace/KMeans-MapReduce-Images-Clustering/data_prep_scripts/ImageCompression/sample_images/img.jpg 
--dst_img /Users/phulocnguyen/Documents/Workspace/KMeans-MapReduce-Images-Clustering/data_prep_scripts/ImageCompression/sample_images/result.jpg