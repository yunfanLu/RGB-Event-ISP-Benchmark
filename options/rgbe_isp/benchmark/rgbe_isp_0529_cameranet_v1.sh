#!/bin/bash

# SBATCH -N 1 # 指定node的数量
# SBATCH -p i64m1tga800u #  i64m1tga40u, i64m1tga800u
# SBATCH --gres=gpu:1 # 需要使用多少GPU，n是需要的数量
# SBATCH --time=7-00:00:00 # 指定任务运⾏的上限时间（我们的HPC为7天
# SBATCH --job-name=rstt-TL-vfi-d1020-4-7-256-pretrained-large-v4
# SBATCH --output=./hpc-eccv-cstvsr/%j-job.out # slurm的输出文件，%j是jobid
# SBATCH --error=./hpc-eccv-cstvsr/%j-error.err # 指定错误输出的格式
# SBATCH --cpus-per-task=8


export CUDA_VISIBLE_DEVICES="0"
export PATH="/hpc2hdd/home/ylu066/miniconda3/bin/":$PATH
export PYTHONPATH="./3rdparty/":$PYTHONPATH
export PYTHONPATH="./":$PYTHONPATH

python ev_rgb_isp/main.py \
  --yaml_file="options/rgbe_isp/benchmark/rgbe_isp_0529_cameranet_v1.yaml" \
  --log_dir="./log/rgbe_isp/benchmark/rgbe_isp_0529_cameranet_v1/" \
  --alsologtostderr=True \
