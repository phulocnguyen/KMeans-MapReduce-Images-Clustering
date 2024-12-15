#!/usr/bin/env bash

# create directories on hdfs
hadoop fs -mkdir -p /KMeans/resources/input/compression/kmeans++
hadoop fs -mkdir -p /KMeans/resources/output/compression/kmeans++

# copy local input files
hadoop fs -put ./resources/input/ImageCompression/kmeans++/points.txt ./resources/input/ImageCompression/kmeans++/clusters.txt /KMeans/resources/input/compression/kmeans++

# specify input parameters
JAR_PATH=./executable_jar/kmeans_mapreduce_v2.jar
MAIN_CLASS=Main
INPUT_FILE_PATH=/KMeans/resources/input/compression/kmeans++/points.txt
STATE_PATH=/KMeans/resources/input/compression/kmeans++/clusters.txt
NUMBER_OF_REDUCERS=3
OUTPUT_DIR=/KMeans/resources/output/compression/kmeans++
DELTA=20.0
MAX_ITERATIONS=10
DISTANCE=euclidean

hadoop jar ${JAR_PATH} ${MAIN_CLASS} --input ${INPUT_FILE_PATH} \
--state ${STATE_PATH} \
--number ${NUMBER_OF_REDUCERS} \
--output ${OUTPUT_DIR} \
--delta ${DELTA} \
--max ${MAX_ITERATIONS} \
--distance ${DISTANCE}

hadoop fs -ls /KMeans/resources/output/compression/kmeans++

hadoop fs -get /KMeans/resources/output/compression/kmeans++/5  ./resources/output/Compression

# execute jar file
LAST_DIR="$(hadoop fs -ls -t -C /KMeans/resources/output/compression/kmeans++ | head -1)"

# print results
hadoop fs -cat "$LAST_DIR/part-r-[0-9][0-9][0-9][0-9][0-9]" | sort --numeric --key 1