#!/bin/bash
nodes=""

while IFS= read -r line
do
  if [ "$nodes" = "" ]; then
    nodes="$line"
  else 
    nodes="$nodes,$line"
  fi
done <"$PBS_NODEFILE"

PROJECT_DIR=$1
mpiexec -ppn 1 bash $PROJECT_DIR/python/osc/mpi.sh $PROJECT_DIR $nodes
