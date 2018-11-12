# Each machine in the cluster will execute this script.
# It initializes the environment and execute the python script to
# start the ps or worker host in the tensorflow cluster.

module load python/2.7 cuda/8.0.44
source activate py27

PROJECT_DIR=$1
cd $PROJECT_DIR

export PYTHONPATH=$PROJECT_DIR
python -u $PROJECT_DIR/python/osc/mpi.py  --nodes=$2 --script=$PROJECT_DIR/python/capsnet_1d/capsnet_1d_dist_train.py --script_params="--model=hippo6 --data_dir=./data/hippo/ --dataset=hippo --summary_dir=./checkpoints/hippo/caps/train_6_9 --file_start=1 --file_end=110 --num_classes=2" --log_file=$PROJECT_DIR/logs/caps_hippo_6_9.log