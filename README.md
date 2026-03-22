# HiFICL: High-Fidelity In-Context Learning for Multimodal Tasks

<p align="center">
<a href="http://arxiv.org/abs/2603.12760">
<img alt="Static Badge" src="https://img.shields.io/badge/arXiv-2603.12760-red"></a>
</p>

**HiFICL (High-Fidelity In-Context Learning)** is a framework that extends [MimIC](https://github.com/Kamichanw/MimIC) to support multiple multimodal models. By integrating lightweight learnable modules into different vision-language models, it demonstrates superior performance across various model architectures.

## Features

- **Multi-Model Support**: Works with idefics, Qwen3-VL, LLaVA-Interleave, and more
- **Flexible Architecture**: Easy to adapt to new multimodal models
- **Multiple Shift Strategies**: Supports various in-context learning methods

## Supported Models

| Subdirectory | Supported Models |
|--------------|------------------|
| `hifi_code_idefics` | idefics-9b, idefics2-8b |
| `hifi-code-qwenvl` | Qwen3-VL-8B-Instruct |
| `hifi-code-llava-interleave` | llava-interleave-qwen-7b |

## Setup

### 1. Create environment
```bash
conda create -y -n hifi python=3.10
conda activate hifi

# Choose the subdirectory you want to use
cd hifi_code_idefics  # or hifi-code-qwenvl / hifi-code-llava-interleave

# Install dependencies (each subdirectory has its own requirements.txt)
pip install -r requirements.txt
```

**Note**: Each subdirectory has its own `requirements.txt` with slightly different `transformers` versions:
- `hifi_code_idefics`: `transformers==4.45.2`
- `hifi-code-qwenvl`: `transformers==4.57.1`
- `hifi-code-llava-interleave`: `transformers==4.57.1`

### 2. Specify the root path of your models and datasets in `src/paths.py`

Edit the paths in `src/paths.py` to match your local setup:
- **Models**: idefics, Qwen2.5-VL, llava-interleave, etc.
- **Datasets**: VQAv2, OK-VQA, COCO, Flickr30k, MME, SEED-bench, etc.

## How to Run

```bash
cd ./script
# Select a bash file to run
bash run_*.sh 
```

## Key Differences from MimIC

This project extends MimIC with the following enhancements:

1. **New Shift Strategy**: Implements a novel approach (see `src/shift_encoder.py`) for better approximating in-context demo effects
2. **Extended Model Support**: Added support for Qwen2.5-VL and LLaVA-Interleave family models

For detailed method description, please refer to our paper.

## Code Structure

### Project Organization
Each subdirectory (`hifi_code_idefics`, `hifi-code-qwenvl`, `hifi-code-llava-interleave`) follows the same structure:

```
├── src/
│   ├── shift_encoder.py   # Core shift mechanism implementation
│   ├── shift_model.py     # Training framework
│   ├── paths.py           # Path configuration
│   ├── utils.py           # Utility functions
│   └── ...
├── scripts/               # Training and evaluation scripts
├── testbed/               # Model implementations
└── assets/                # Figures and resources
```

### Key Files

#### [`shift_encoder.py`](./src/shift_encoder.py)
Implements the HiFICL attention heads (`AttnApproximator`) and shift mechanisms. The core innovation includes:
- `AttnApproximator`: Attention-based approximation for in-context demo effects
- `AttnFFNShift`: Vector-based shift method (similar to LIVE)
- `ShiftStrategy`: Configurable strategies `MULTI_HEAD`

#### [`shift_model.py`](./src/shift_model.py)
Training framework implementation with configurable loss strategies:
- `Strategy.LAYER_WISE_MSE`: Layer-wise alignment loss
- `Strategy.LM_LOSS`: Language modeling loss
- `Strategy.LOGITS_KL_DIV`: KL divergence loss

## Customization

### Add New Datasets
1. Add dataset path to `src/paths.py`
2. Create a new dataset class in `src/dataset_utils/`
3. Inherit from `src.dataset_utils.interface.DatasetBase`
4. Implement required abstract methods

### Add New Models
1. Add model path to `src/paths.py`
2. Create model implementation in `testbed/models/`
3. Update `build_models` in `src/utils.py`
4. Implement corresponding model hooks in `shift_encoder.py`

## Contact

**Yuhang Liu**
- 📧 [202422090537@std.uestc.edu.cn](mailto:202422090537@std.uestc.edu.cn) (Academic)
- 📧 [2292261265@qq.com](mailto:2292261265@qq.com) (Personal)

## Recommended Citation
```
@misc{li2026hificlhighfidelityincontextlearning,
      title={HIFICL: High-Fidelity In-Context Learning for Multimodal Tasks}, 
      author={Xiaoyu Li and Yuhang Liu and Zheng Luo and Xuanshuo Kang and Fangqi Lou and Xiaohua Wu and Zihan Xiong},
      year={2026},
      eprint={2603.12760},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.12760}, 
}
```
