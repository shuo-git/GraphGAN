export PYTHONPATH=/data/private/ws/projects/GraphGAN/ShuoGraphGAN:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=8

CODE=/data/private/ws/projects/GraphGAN/ShuoGraphGAN/graphGAN
DATA=/data/private/ws/projects/GraphGAN/ShuoGraphGAN/data/ml-1m
log_dir=./train
mkdir -p ${log_dir}
python ${CODE}/bin/trainer.py --data_dir ${DATA} --log_dir ${log_dir}
