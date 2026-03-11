from functools import lru_cache
import itertools
from typing import Any, Dict, List, Optional, Union
from PIL.Image import Image
import warnings
# from modelscope import LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers import (
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
)

from .model_base import ModelBase


HF_LLAVA_NEXT = {
    "llava-v1.6": ["llava-v1.6-mistral-7b-hf"],
    "llava-next": ["llava-next"],
}


class LLaVaNext(ModelBase):
    def __init__(
        self,
        model_root,
        processor_class=LlavaNextProcessor,
        model_class=LlavaNextForConditionalGeneration,
        processor_args=None,
        model_args=None,
        **common_args,
    ):

        processor_args = (
            processor_args if processor_args else dict()
        )
        
        # 提取 image_grid_pinpoints 以便在 processor 创建后设置
        image_grid_pinpoints = processor_args.pop("image_grid_pinpoints", None)
        
        super().__init__(
            model_root=model_root,
            processor_class=processor_class,
            model_class=model_class,
            support_models=list(name for v in HF_LLAVA_NEXT.values() for name in v),
            processor_args=processor_args,
            model_args=model_args,
            **common_args,
        )
        self.processor.patch_size = self.config.vision_config.patch_size
        self.processor.vision_feature_select_strategy = (
            self.config.vision_feature_select_strategy
        )
        
        # 在 processor 和 model 创建后设置 image_grid_pinpoints 来控制分片
        # 例如: {"image_grid_pinpoints": [[336, 336]]} 禁用分片
        # 或者: {"image_grid_pinpoints": [[336, 672], [672, 336], ...]} 启用分片
        if image_grid_pinpoints is not None:
            # 更新 processor 配置
            self.processor.image_processor.image_grid_pinpoints = image_grid_pinpoints
            # 更新 model 配置（关键！get_image_features 使用 model.config.image_grid_pinpoints）
            self.model.config.image_grid_pinpoints = image_grid_pinpoints

    @property
    def default_prompt_template(self):
        # LLaVA Next 使用与 LLaVA 1.5 相同的模板
        # fmt: off
        return (
                "{% if messages[0]['role'].lower() in ['instruction', 'system'] %}"
                    "{{ messages[0]['content'] + '\n' }}"
                    "{% set messages = messages[1:] %}"
                "{% endif %}"
                "{% set first_role = messages[0]['role'] %}"
                "{% set ns = namespace(generation_role='ASSISTANT') %}"
                "{% for message in messages %}"
                    "{% if loop.last or loop.nextitem['role'] == first_role %}"
                        "{% set ns.generation_role = message['role'] %}"
                    "{% endif %}"
                    "{{ message['role'].upper() }}"
                    "{% if 'content' in message %}"
                        "{{ ': ' }}"
                        "{# Render all images first #}"
                        "{% for line in message['content'] | selectattr('type', 'equalto', 'image')%}"
                            "{{ '<image>\n' }}"
                        "{% endfor %}"
                        "{# Render all text next #}"
                        "{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}"
                            "{% if message['role'] != ns.generation_role %}"
                                "{{ content['text'] + ' ' }}"
                            "{% else %}"
                                "{% generation %}"
                                    "{{ content['text'] + ' '}}"
                                "{% endgeneration %}"
                            "{% endif %}"
                        "{% endfor %}"
                    "{% else %}"
                        "{{ ':' }}"
                    "{% endif %}"
                "{% endfor %}"
                "{% if add_generation_prompt %}"
                    "{{ ns.generation_role.upper() + ':' }}"
                "{% endif %}"
            )
        # fmt: on

    def process_input(
        self,
        images: Union[List[Image], List[List[Image]]],
        text: Union[
            List[Union[str, Dict[str, Any]]], List[List[Union[str, Dict[str, Any]]]]
        ],
        prompt_template: Optional[str] = None,
        **kwargs,
    ):
        """
        Processes text and image inputs for the model.
        """
        if isinstance(text[0], dict) or (
            isinstance(text[0], list) and isinstance(text[0][0], dict)
        ):
            text = self.apply_prompt_template(text, prompt_template=prompt_template)  # type: ignore[arg-type]
        
        if isinstance(images[0], list):
            # llava doesn't support images with type List[List[Image]]
            images = list(itertools.chain(*images))

        return self.processor(
            images=images,
            text=text,
            padding=kwargs.pop("padding", True),
            return_tensors=kwargs.pop("return_tensors", "pt"),
            **kwargs,
        )
