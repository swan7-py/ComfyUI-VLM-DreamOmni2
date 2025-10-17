import torch
import numpy as np
from PIL import Image
import os
import folder_paths
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor



class VLMImageEditingPrompt:
    """VLM图像编辑提示词生成节点 - 支持 fp16 / int8 / int4 量化"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_1": ("IMAGE",),
                "task_type": (["edit", "generate"], {"default": "edit"}),
                "instruction": ("STRING", {
                    "multiline": True,
                    "default": "Make the first image have the same style as the second image"
                }),
                "quantization": (["fp16", "int8", "int4"], {"default": "int4"}),  
            },
            "optional": {
                "image_2": ("IMAGE",),
                "max_new_tokens": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 2048,
                    "step": 64
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("enhanced_prompt",)
    FUNCTION = "generate_enhanced_prompt"
    CATEGORY = "DreamOmni2"
    DESCRIPTION = "使用VLM模型根据输入图像和指令生成增强的编辑提示词（支持 fp16/int8/int4 量化）"

    def __init__(self):
        self.vlm_model = None
        self.processor = None
        self.loaded_quant = None  # 记录当前加载的量化类型，避免重复加载
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vlm_model_path = os.path.join(folder_paths.models_dir, "vlm_model")

    def load_vlm_model(self, quantization):
        """根据量化类型加载模型，避免重复加载相同配置"""
        if self.vlm_model is not None and self.loaded_quant == quantization:
            return  

        print(f"Loading VLM model from: {self.vlm_model_path} with quantization: {quantization}")
        try:
            if not os.path.exists(self.vlm_model_path):
                raise Exception(f"VLM模型路径不存在: {self.vlm_model_path}")

            from transformers import BitsAndBytesConfig

            # 默认 dtype
            torch_dtype = torch.float16
            quant_config = None

            if quantization == "int8":
                quant_config = BitsAndBytesConfig(load_in_8bit=True)
            elif quantization == "int4":
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )

            self.vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.vlm_model_path,
                torch_dtype=torch_dtype,
                quantization_config=quant_config,
                device_map="auto",
                trust_remote_code=True,
            )
            self.processor = AutoProcessor.from_pretrained(
                self.vlm_model_path,
                trust_remote_code=True,
            )

            self.loaded_quant = quantization
            print(f"VLM model loaded successfully with {quantization} quantization")

        except Exception as e:
            print(f"Error loading VLM model with {quantization}: {e}")
            raise e

    def tensor_to_pil(self, tensor):
        if len(tensor.shape) == 4:
            tensor = tensor[0]
        tensor = tensor.clamp(0, 1)
        numpy_image = (tensor.numpy() * 255).astype(np.uint8)
        if numpy_image.shape[-1] == 1:
            numpy_image = numpy_image[:, :, 0]
        return Image.fromarray(numpy_image)

    def extract_gen_content(self, text):
        return text[6:-7].strip()

    def generate_enhanced_prompt(self, image_1, task_type, instruction, quantization="fp16", image_2=None, max_new_tokens=1024):
        try:
            self.load_vlm_model(quantization) 

            if self.vlm_model is None or self.processor is None:
                raise Exception("VLM模型加载失败，请检查模型路径")

            pil_image_1 = self.tensor_to_pil(image_1)
            content = []
            content = [{"type": "image", "image": pil_image_1}]

            if image_2 is not None:
                pil_image_2 = self.tensor_to_pil(image_2)
                content.append({"type": "image", "image": pil_image_2})

            if task_type == "edit":
                enhanced_instruction = instruction + " It is editing task."
            else: 
                enhanced_instruction = instruction + " It is generation task."
            
            print(f"{enhanced_instruction}")
            content.append({"type": "text", "text": enhanced_instruction})
            
            messages = [{"role": "user", "content": content}]

            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            image_inputs = []
            for message in messages:
                for item in message["content"]:
                    if item["type"] == "image":
                        image_inputs.append(item["image"])

            inputs = self.processor(text=[text], images=image_inputs, padding=True, return_tensors="pt")
            inputs = inputs.to(self.device)

            with torch.no_grad():
                generated_ids = self.vlm_model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )

            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
            enhanced_prompt = self.extract_gen_content(output_text)

            print(f"[{quantization.upper()}] VLM生成的增强提示词: {enhanced_prompt}")
            return (enhanced_prompt,)

        except Exception as e:
            print(f"VLM生成提示词时出错 ({quantization}): {e}")
            backup_prompt = f"{instruction} - reference style from second image" if image_2 is not None else instruction
            return (backup_prompt,)

NODE_CLASS_MAPPINGS = {
    "VLMImageEditingPrompt": VLMImageEditingPrompt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VLMImageEditingPrompt": "VLMImageEditingPrompt",
}
