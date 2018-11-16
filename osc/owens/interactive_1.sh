#!/bin/bash

# This script can be taken as the test script in interactive mode.
# It distributes a task to each machine in the cluster with mpiexec.
# It should be called with the project dir as its 1st argument.

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
echo "nodes: $nodes"
mpiexec -ppn 1 bash $PROJECT_DIR/osc/owens/mpi_1.sh $PROJECT_DIR $nodes
# perf-report --np=5 bash $PROJECT_DIR/osc/owens/mpi.sh $PROJECT_DIR $nodes
