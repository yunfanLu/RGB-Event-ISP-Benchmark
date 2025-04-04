#!/bin/bash

#SBATCH -N 1 # 指定node的数量
#SBATCH -p i64m1tga800u #  i64m1tga40u, i64m1tga800u
#SBATCH --gres=gpu:1 # 需要使用多少GPU，n是需要的数量
#SBATCH --time=7-00:00:00 # 指定任务运⾏的上限时间（我们的HPC为7天
#SBATCH --job-name=rstt-TL-vfi-d1020-4-7-256-pretrained-large-v4
#SBATCH --output=./hpc-eisp/%j-job.out # slurm的输出文件，%j是jobid
#SBATCH --error=./hpc-eisp/%j-error.err # 指定错误输出的格式
#SBATCH --cpus-per-task=1


export CUDA_VISIBLE_DEVICES="0"
export PATH="/hpc2hdd/home/ylu066/miniconda3/bin/":$PATH
export PYTHONPATH="./3rdparty/":$PYTHONPATH
export PYTHONPATH="./":$PYTHONPATH

# python tools/22-Color-Events-Eval-in-Each-Video/color_events_isp_eval_in_each_video.py \
#   --yaml_file="options/rgbe_isp/benchmark/rgbe_isp_0529_unet_v1.yaml" \
#   --log_dir="./log/rgbe_isp/benchmark/rgbe_isp_0529_unet_v1-test/" \
#   --alsologtostderr=True \
#   --RESUME_PATH="././log/rgbe_isp/benchmark/rgbe_isp_0529_unet_v1/checkpoint-040.pth.tar" \
#   --TEST_ONLY=True \
#   --TEST_TAG="unet_v1" \
#   --VISUALIZE=True

python tools/22-Color-Events-Eval-in-Each-Video/color_events_isp_eval_in_each_video.py \
  --yaml_file="options/rgbe_isp/benchmark/rgbe_isp_0529_unet_swin_v1.yaml" \
  --log_dir="./log/rgbe_isp/benchmark/rgbe_isp_0529_unet_swin_v1-test-epoch25/" \
  --alsologtostderr=True \
  --RESUME_PATH="./log/rgbe_isp/benchmark/rgbe_isp_0529_unet_swin_v1/checkpoint-025.pth.tar" \
  --TEST_ONLY=True \
  --TEST_TAG="swin_v1" \
  --VISUALIZE=True

