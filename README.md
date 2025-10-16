# ComfyUI-VLM-DreamOmni2
DreamOmni2中VLM在ComfyUI中的复现，支持int4,int8量化；配合loras可完成原项目的复现

本项目所使用的 VLM 模型及 LoRA 权重均来自开源项目 **[DreamOmni2](https://github.com/dvlab-research/DreamOmni2)**（由 DVLab 研究团队发布）。  
我们对原作者的卓越工作表示由衷感谢！本节点能够在 ComfyUI 中进行模块化复现，以及未来拓展，**并非官方实现**，请遵守原项目的使用条款。

## 📦 安装指南

### 1. 克隆本仓库到 ComfyUI

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/your-username/ComfyUI-VLM-Quantized-Prompt.git
pip install -r ComfyUI-VLM-DreamOmni2/requirements.txt  [可选]
```

### 2、下载 VLM 模型
将 DreamOmni2 的 VLM_models（注意原文件夹名称"-"要改为"_"） 下载至 ComfyUI/models/
https://huggingface.co/xiabs/DreamOmni2

### 3、下载 LoRA 权重（用于完整 DreamOmni2 流程）
| 用途             | 下载链接                                                                                                                                     | 保存路径与文件名                                      |
|------------------|----------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------|
| 编辑任务 LoRA    | [pytorch_lora_weights.safetensors (edit)](https://huggingface.co/xiabs/DreamOmni2/resolve/main/edit_lora/pytorch_lora_weights.safetensors)    | `ComfyUI/models/loras/DreamOmni2_edit.safetensors`  |
| 生成任务 LoRA    | [pytorch_lora_weights.safetensors (gen)](https://huggingface.co/xiabs/DreamOmni2/resolve/main/gen_lora/pytorch_lora_weights.safetensors)      | `ComfyUI/models/loras/DreamOmni2_gen.safetensors`   |

### 4、工作流在example_workflow中
如果你已经安装了nunchaku，还可以使用nunchaku进行模型部分的加速
<img width="2783" height="1662" alt="workflow (1)" src="https://github.com/user-attachments/assets/dbdecf27-a571-4e3c-9c90-01acece2826e" />
gen模式
<img width="3102" height="1925" alt="workflow (2)" src="https://github.com/user-attachments/assets/fad8b3c6-df67-4163-91a9-88292bb61198" />
edit模式
