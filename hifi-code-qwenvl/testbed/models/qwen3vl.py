from functools import lru_cache
import itertools
from typing import Any, Dict, List, Optional, Union
from PIL.Image import Image
import warnings
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
)

from .model_base import ModelBase


class Qwen3VL(ModelBase):
    def __init__(
        self,
        model_root,
        processor_class=AutoProcessor,
        model_class=Qwen3VLForConditionalGeneration,
        processor_args=None,
        model_args=None,
        **common_args,
    ):
        super().__init__(
            model_root=model_root,
            processor_class=processor_class,
            model_class=model_class,
            processor_args=processor_args,
            model_args=model_args,
            **common_args,
        )

        # Set chat template for Qwen3-VL
        if hasattr(self.processor, 'chat_template') and self.processor.chat_template is None:
            # Use a simple chat template that includes image placeholders
            self.processor.chat_template = self.default_prompt_template

    @property
    def default_prompt_template(self):
        @lru_cache
        def warn_once(msg):
            warnings.warn(msg)

        if self.model_name.startswith("Qwen3-VL"):
            # Use a prompt template that normalizes non-standard role names
            # into `user` / `assistant` and supports image placeholders.
            return (
                "{% if messages[0]['role'] == 'system' %}"
                    "{{ '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' }}"
                    "{% set messages = messages[1:] %}"
                "{% endif %}"
                "{% for message in messages %}"
                    "{{ '<|im_start|>' + message['role'] + '\n' }}"
                    "{% for content in message['content'] %}"
                        "{% if content['type'] == 'image' %}"
                            "{{ '<|vision_start|><|image_pad|><|vision_end|>' }}"
                        "{% elif content['type'] == 'text' %}"
                            "{% if message['role'] == 'assistant' %}"
                                "{% generation %}"
                                    # 重点：这里加回了 + ' '，适配英文分词
                                    "{{ content['text'] + ' ' }}"
                                "{% endgeneration %}"
                            "{% else %}"
                                # 重点：User 部分也加回了 + ' '
                                "{{ content['text'] + ' ' }}"
                            "{% endif %}"
                        "{% endif %}"
                    "{% endfor %}"
                    "{{ '<|im_end|>\n' }}"
                "{% endfor %}"
                "{% if add_generation_prompt %}"
                    "{{ '<|im_start|>assistant\n' }}"
                "{% endif %}"
            )
        else:
            warn_once(
                f"The model {self.model_name} is not in official Qwen3-VL collections. "
                "Please either customize your own prompt template for this model, "
                "or set `model_name` to select a default prompt template."
            )
            return super().default_prompt_template

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

        Args:
            images (Union[List[Image], List[List[Image]]]):
                A list of images or a list of lists of images. For unbatched input, this should be a single-level list
                of images. For batched input, this should be a nested list where each inner list represents a batch of images.
                Each image should be an instance of the `Image` class.

            text (Union[List[Union[str, Dict[str, Any]]], List[List[Union[str, Dict[str, Any]]]]]):
                A list of texts or a list of lists of texts. For unbatched input, this should be a single-level list
                where each item is either a string or a dictionary. For batched input, this should be a nested list
                (list of lists) where each inner list represents a batch of texts. Dictionaries can follow the
                transformers' conversation format, with keys like "role" and "content".

            prompt_template (str, optional):
                A Jinja template which will be used to convert lists of messages in a chat into a tokenizable string.

            **kwargs:
                Additional keyword arguments passed to the `processor`.

        Returns:
            The output of the `processor` function, which is the processed input ready for the model.
        """
        # Handle different text formats
        if isinstance(text[0], dict) or (
            isinstance(text[0], list) and isinstance(text[0][0], dict)
        ):
            # Text is in conversation format, apply prompt template
            print(f"Qwen3VL: Applying prompt template to conversation messages")
            text = self.apply_prompt_template(
                text,
                prompt_template=prompt_template,
                add_generation_prompt=True,
            )  # type: ignore[arg-type]
        elif isinstance(text[0], str) and images:
            # Text is already formatted strings, assume placeholders are already in place
            # from the chat template. If not, we might need to handle this case.
            print(f"Qwen3VL: Using pre-formatted text strings with {len(images)} images")
            # Check if placeholders are present and show sample
            placeholders_present = any(
                "<|vision_start|>" in t or "<|image_pad|>" in t or "<|vision_end|>" in t
                for t in text if isinstance(t, str)
            )
            print(f"Placeholders present: {placeholders_present}, Sample text: {text[0][:150] if text and len(text) > 0 and isinstance(text[0], str) else 'Non-string'}")
            if not placeholders_present:
                print("Warning: No image placeholders found in text, images may not be processed correctly")

        if isinstance(images[0], list):
            # Qwen3-VL may support batched images, flatten if needed
            images = list(itertools.chain(*images))

        # For Qwen3-VL, we need to ensure proper image processing
        # The processor expects text with image placeholders and corresponding images
        try:
            # Check if text contains image placeholders
            text_has_placeholders = any(
                "<|vision_start|>" in t or "<|image_pad|>" in t or "<|vision_end|>" in t
                for t in text if isinstance(t, str)
            )

            # Debug: show text content
            if images and not text_has_placeholders:
                print(f"Qwen3VL: Skipping image processing - text has no placeholders but {len(images)} images provided")
                print(f"Sample text: {text[0][:200] if text and len(text) > 0 and isinstance(text[0], str) else 'Non-string or empty'}")
                return self.processor(
                    text=text,
                    padding=kwargs.pop("padding", True),
                    return_tensors=kwargs.pop("return_tensors", "pt"),
                    **kwargs,
                )
            elif not images:
                # Text-only processing
                return self.processor(
                    text=text,
                    padding=kwargs.pop("padding", True),
                    return_tensors=kwargs.pop("return_tensors", "pt"),
                    **kwargs,
                )
            else:
                # Normal image processing
                return self.processor(
                    images=images,
                    text=text,
                    padding=kwargs.pop("padding", True),
                    return_tensors=kwargs.pop("return_tensors", "pt"),
                    **kwargs,
                )
        except Exception as e:
            # If processing fails, try without images first to debug
            print(f"Qwen3VL processing failed: {e}")
            print(f"Text input sample: {text[0] if text and len(text) > 0 else 'None'}")
            print(f"Images count: {len(images) if images else 0}")
            print(f"Text has placeholders: {any('<|vision_start|>' in t or '<|image_pad|>' in t or '<|vision_end|>' in t for t in text if isinstance(t, str))}")
            try:
                # Try text-only fallback
                return self.processor(
                    text=text,
                    padding=kwargs.pop("padding", True),
                    return_tensors=kwargs.pop("return_tensors", "pt"),
                    **kwargs,
                )
            except Exception as fallback_e:
                print(f"Text-only fallback also failed: {fallback_e}")
                raise e
