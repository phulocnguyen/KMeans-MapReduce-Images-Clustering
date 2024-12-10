#!/usr/bin/env bash

# create directories on hdfs
hadoop fs -mkdir -p /KMeans/resources/input/cluster
hadoop fs -mkdir -p /KMeans/resources/output/cluster

# copy local input files
hadoop fs -put ./resources/input/ImageClustering/points.txt ./resources/input/ImageClustering/clusters.txt /KMeans/resources/input/cluster
# remove output files if any
hadoop fs -rm -r -f /KMeans/resources/output/cluster/*

# specify input parameters
JAR_PATH=./executable_jar/kmeans_mapreduce.jar
MAIN_CLASS=Main
INPUT_FILE_PATH=/KMeans/resources/input/cluster/points.txt
STATE_PATH=/KMeans/resources/input/cluster/clusters.txt
NUMBER_OF_REDUCERS=3
OUTPUT_DIR=/KMeans/resources/output/cluster
DELTA=10.0
MAX_ITERATIONS=10
DISTANCE=eucl

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


hadoop fs -get /KMeans/resources/output/cluster/2 ./resources/output