# Event-RGB ISP: Benchmark 🎉

Welcome to the **RGB-Event ISP Benchmark**! 🌈✨ This toolkit provides a comprehensive evaluation framework for Image Signal Processing (ISP) methods leveraging both RGB and event data. Designed for researchers and developers, it supports multiple ISP categories and offers cute visualization tools! 🚀

---

## 🌟 Features
- **Multi-Category ISP Evaluation** 📊
  Supports 4 ISP method types with 🤖 **AI-friendly** training pipelines:
  - **Full Pipeline ISP** (PyNet, MV-ISPNet, InvertISP) 🛠️
  - **Stage-wise ISP** (CameraNet, AWNet) 🎯
  - **Image Enhancement Networks** (UNet, Swin-Transformer) 🎨
  - **Event Fusion Methods** (EV-UNet, eSL-Net) ⚡

- **Rich Metrics** 📏
  Evaluates models on:
  - **Resource Consumption**: Params (M), GFLOPS, Inference Time ⏱️
  - **Image Quality**: PSNR / SSIM / L1 for indoor & outdoor scenes 🏞️🏠

- **Pixel-Aligned Dataset** 📸
  Includes **3,373 high-res RAW-event pairs** with HVS sensor alignment (see [paper](https://arxiv.org/abs/xxx) for details)!

---

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/your_username/rgb-event-isp-benchmark.git
cd rgb-event-isp-benchmark
pip install -r requirements.txt  # 🐍 Python 3.8+ required
```

### Run a Benchmark Example 🧪

All configs are in the `options` folder. For example, to run the benchmark with AWNet on this dataset:

```bash
python ev_rgb_isp/main.py \
  --yaml_file="options/rgbe_isp/benchmark/rgbe_isp_0529_awnet_v1.yaml" \
  --log_dir="./log/rgbe_isp/benchmark/rgbe_isp_0529_awnet_v1/" \
  --alsologtostderr=True
```
*✨ Pro Tip:* Use `tensorboard --logdir=your_log_dir` to monitor training! 📈

---

## 📚 Benchmark Details

### 🏆 Evaluated Methods
| Category                     | Models                          | Highlights                          |
|------------------------------|---------------------------------|-------------------------------------|
| **Full Pipeline ISP** 🚀      | PyNet, MV-ISPNet, InvertISP     | End-to-end RAW→RGB conversion       |
| **Stage-wise ISP** 🧩         | CameraNet, AWNet                | Modular design for ISP subtasks     |
| **Enhancement Networks** 🌟  | UNet, Swin-Transformer         | Adapted from image restoration tasks|
| **Event Fusion** ⚡          | EV-UNet, eSL-Net               | First exploration of event-guided ISP! |

### 🛠️ Training Config
- **Hardware**: Single NVIDIA A40 GPU (48GB VRAM) 💻
- **Augmentation**: Random crop (1024×1024) + rotation 🌪️
- **Optimization**: Adam w/ lr=1e-4, 50 epochs ⏳
- **Batch Size**: 1 (due to high-res patches) 📦

---



## 🎨 Visualization
![Outdoor Results](images/R2-Outdoor.jpg)
*Figure 1: Outdoor scene comparisons*

![Indoor Results](images/R3-Compre-Vis-Indoor-Release.jpeg)
*Figure 2: Indoor scene comparisons*

---

## 📜 Citation
If this work helps your research, please cite:
```bibtex
@inproceedings{lu2025rgb,
  title={RGB-Event ISP: The Dataset and Benchmark},
  author={Yunfan Lu and Yanlin Qian and Ziyang Rao and Junren Xiao and Liming Chen and Hui Xiong},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
```

---

Made with ❤️ by Yunfan LU | 2024
Questions? → `ylu066@connect.hkust-gz.edu.cn` 📧
*Keep calm and process images!* 📷✨
