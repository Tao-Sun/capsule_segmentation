# Each machine in the cluster will execute this script.
# It initializes the environment and execute the python script to
# start the ps or worker host in the tensorflow cluster.

module load python/2.7 cuda/8.0.44
source activate py27

PROJECT_DIR=$1
cd $PROJECT_DIR

export PYTHONPATH=$PROJECT_DIR
python -u $PROJECT_DIR/python/osc/mpi.py  --nodes=$2 --script=$PROJECT_DIR/python/capsnet_1d/capsnet_1d_dist_train.py --script_params="--model=mnist5 --data_dir=$PROJECT_DIR/data/mnist/train_all/ --dataset=mnist --summary_dir=$PROJECT_DIR/checkpoints/mnist/caps/train_all_5 --file_start=0 --file_end=4" --log_file=$PROJECT_DIR/logs/caps_mnist_all_5_1