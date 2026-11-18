# HoloGen Pretrained Models Registry & Licensing

Per Rule 6 of the HoloGen build specification, no model weights are committed to git. All models are automatically downloaded on first run into `models/cache/huggingface/` (or `models/cache/torch/`) and verified for license compliance before adoption.

## Adopted Models

| Model Key | Hugging Face / Hub ID | Architecture | Output Type | Parameters | License | Source URL | Weights Cache Location |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `depth_anything_v2_small` | `depth-anything/Depth-Anything-V2-Small-hf` | DINOv2 ViT-Small + DPT Head | Relative affine-invariant disparity | 24.8M | **Apache-2.0** | https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf | `models/cache/huggingface/` |
| `dpt_hybrid_midas` | `Intel/dpt-hybrid-midas` | ResNet50 + ViT-Base Hybrid DPT | Relative affine-invariant disparity | 123.0M | **MIT** | https://huggingface.co/Intel/dpt-hybrid-midas | `models/cache/huggingface/` |
| `depth_anything_v2_metric_indoor_small` | `depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf` | DINOv2 ViT-Small + Metric DPT Head (Hypersim/NYU) | Metric depth (meters) | 24.8M | **Apache-2.0** | https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf | `models/cache/huggingface/` |

## Licensing Notes

1. **Depth Anything V2 Small (`depth-anything/Depth-Anything-V2-Small-hf`)**:
   - The **Small** (`ViT-S`) checkpoints of Depth Anything V2 (both relative and metric variants) are explicitly released by the authors under the permissive **Apache-2.0 License** (whereas the Base, Large, and Giant variants use CC-BY-NC-4.0 because they depend on DINOv2-Base/Large/Giant). Adopting the Small variants guarantees both permissive Apache-2.0 licensing and compatibility with 2 GB VRAM GPUs (NVIDIA GeForce MX330).
2. **Intel DPT-Hybrid MiDaS (`Intel/dpt-hybrid-midas`)**:
   - Released by Intel ISL under the permissive **MIT License**.
3. **Excluded Candidates**:
   - `ZoeDepth` and `Metric3D-Large` (`ViT-Large` / `ConvNeXt-Large`) exceed the 2.0 GB VRAM ceiling of the target NVIDIA GeForce MX330 GPU and are therefore excluded in favor of `Depth-Anything-V2-Metric-Indoor-Small-hf`.
