from abc import ABC, abstractmethod
from PIL import Image
import os
import torch
import base64
from io import BytesIO
from openai import OpenAI
from transformers import AutoProcessor, AutoModelForVision2Seq, AwqConfig, Qwen2VLForConditionalGeneration
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
import unstructured_client
from unstructured_client.models import shared
import google.generativeai as genai
from qwen_vl_utils import process_vision_info


class OcrProvider(ABC):
    @abstractmethod
    def ocr(self, doc_image: Image.Image, doc_hash: str) -> str | None:
        pass


OCR_PROMPT = "Extract all the text from this image and explain the non-textual elements, preserving structure as much as possible."


class GeminiOcrProvider(OcrProvider):
    def __init__(self):
        self.gemini_model = genai.GenerativeModel('gemini-1.5-flash-8b')
        self.openai_client = None

    def ocr(self, doc_image: Image.Image, doc_hash: str) -> str | None:
        try:
            return self._ocr_gemini_once(doc_image)
        except ValueError as e:
            if 'encoding error' in str(e):
                doc_image = doc_image.resize((doc_image.width // 2, doc_image.height // 2))
                try:
                    return self._ocr_gemini_once(doc_image)
                except ValueError as e:
                    print(f"Encoding failed even at reduced resolution: {doc_image.size}")
            elif 'copyright' in str(e) or 'blocked' in str(e):
                print(f"OCR for {doc_hash} failed with Flash, trying GPT-4V (error was {str(e)})")
                return self._ocr_gpt4v(doc_image)
        except Exception as e:
            print(f"Unexpected error, skipping document {doc_hash}: {str(e)}")
        return None

    def _ocr_gemini_once(self, doc_image) -> str:
        response = self.gemini_model.generate_content(
            [OCR_PROMPT, doc_image],
            generation_config=genai.types.GenerationConfig(temperature=0, max_output_tokens=2048)
        )
        return response.text

    def _ocr_gpt4v(self, doc_image) -> str:
        if not self.openai_client:
            self.openai_client = OpenAI()
        # Convert PIL image to base64
        buffered = BytesIO()
        doc_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        response = self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": OCR_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_str}"
                            }
                        }
                    ]
                }
            ],
        )
        
        return response.choices[0].message.content


class LlamaOcrProvider(OcrProvider):
    def __init__(self):
        self.llama_parser = LlamaParse(api_key=os.environ.get('LLAMA_CLOUD_API_KEY'),
                                       result_type="markdown")

    def ocr(self, doc_image: Image.Image, doc_hash: str) -> str | None:
        filename = "/tmp/ocr_llama.png"
        doc_image.save(filename)

        file_extractor = {".png": self.llama_parser}
        L = SimpleDirectoryReader(input_files=[filename], file_extractor=file_extractor).load_data()
        if not L:
            print(f"No text extracted from {doc_hash}")
            return None
        return L[0].get_content()


class UnstructuredOcrProvider(OcrProvider):
    def __init__(self):
        self.u_client = unstructured_client.UnstructuredClient(
            api_key_auth=os.getenv("UNSTRUCTURED_API_KEY"),
            server_url=os.getenv("UNSTRUCTURED_API_URL")
        )

    def ocr(self, doc_image: Image.Image, doc_hash: str) -> str:
        filename = "/tmp/ocr_unstructured.png"
        doc_image.save(filename)
        language = 'fr' if 'tabfquad' in self.current_dataset_name or 'shift' in self.current_dataset_name else 'eng'
        req = {
            "partition_parameters": {
                "files": {
                    "content": open(filename, "rb"),
                    "file_name": filename,
                },
                "strategy": shared.Strategy.HI_RES,
                "languages": [language],
            }
        }
        res = self.u_client.general.partition(request=req)
        return '\n\n'.join(e['text'] for e in res.elements)


class Idefics2OcrProvider(OcrProvider):
    def __init__(self, device):
        self.device = device
        self.idefics2_processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b")
        quantization_config = AwqConfig(
            bits=4,
            fuse_max_seq_len=4096,
            modules_to_fuse={
                "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "mlp": ["gate_proj", "up_proj", "down_proj"],
                "layernorm": ["input_layernorm", "post_attention_layernorm", "norm"],
                "use_alibi": False,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "hidden_size": 4096,
            }
        )
        self.idefics2_model = AutoModelForVision2Seq.from_pretrained(
            "HuggingFaceM4/idefics2-8b-AWQ",
            torch_dtype=torch.float16,
            quantization_config=quantization_config,
        ).to(self.device)

    def ocr(self, doc_image: Image.Image, doc_hash: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text",
                     "text": "Extract all the text from this image, preserving structure as much as possible."},
                ]
            }
        ]
        prompt = self.idefics2_processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.idefics2_processor(text=prompt, images=[doc_image], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self.idefics2_model.generate(**inputs, max_new_tokens=500)
        generated_text = self.idefics2_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return generated_text


class Qwen2OcrProvider(OcrProvider):
    def __init__(self, device):
        self.device = device
        quantization_config = AwqConfig(bits=4, group_size=128, zero_point=True, modules_to_not_convert=["lm_head"])
        self.qwen2_model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct-AWQ", torch_dtype="auto", device_map="auto"
        )
        self.qwen2_processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct-AWQ")

    def ocr(self, doc_image: Image.Image, doc_hash: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image",
                     "image": doc_image},
                    {"type": "text",
                     "text": "Extract all the text from this image and explain the non-textual elements, preserving structure as much as possible."},
                ],
            }
        ]

        text = self.qwen2_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.qwen2_processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        with torch.no_grad():
            generated_ids = self.qwen2_model.generate(**inputs, max_new_tokens=2048)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.qwen2_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return output_text
