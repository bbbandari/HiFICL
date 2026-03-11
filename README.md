# HiFICL: High-Fidelity In-Context Learning for Multimodal Tasks

<p align="center">
<a href="https://arxiv.org/abs/XXXX.XXXXX">
<img alt="Static Badge" src="https://img.shields.io/badge/arXiv-XXXX.XXXXX-red"></a>
</p>

<img src="./assets/overview.png" alt="overview figure" style="display: block; margin: 0 auto; width: 90%;" />

**HiFICL (High-Fidelity In-Context Learning)** is a framework that extends [MimIC](https://github.com/xxx/mimic) to support multiple multimodal models. By integrating lightweight learnable modules into different vision-language models, it demonstrates superior performance across various model architectures.

## Features

- **Multi-Model Support**: Works with idefics, Qwen2.5-VL, LLaVA-Interleave, and more
- **Flexible Architecture**: Easy to adapt to new multimodal models
- **Multiple Shift Strategies**: Supports various in-context learning methods

## Supported Models

| Subdirectory | Supported Models |
|--------------|------------------|
| `hifi_code_idefics` | idefics-9b, idefics2-8b |
| `hifi-code-qwenvl` | Qwen2.5-VL-7B-Instruct, Qwen3-VL-8B-Instruct |
| `hifi-code-llava-interleave` | llava-interleave-qwen-7b, llava-v1.6-mistral-7b, llama3-llava-next-8b, llava-onevision-qwen2-7b |

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
3. **Improved Architecture**: Enhanced shift encoder with NLICV_MLP_STATIC_SCALE and multi-head attention mechanisms

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
- `ShiftStrategy`: Configurable strategies including `NLICV_MLP_STATIC_SCALE` and `MULTI_HEAD`

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

# Recommended Citation
```
@InProceedings{YourName_2026_CVPR,
    author    = {Your Name},
    title     = {HiFICL: High-Fidelity In-Context Learning for Multimodal Tasks},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {XXXXX-XXXXX}
}
```
