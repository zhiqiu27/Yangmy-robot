import cv2
import numpy as np
import supervision as sv
import torch
from autodistill.detection import CaptionOntology, DetectionBaseModel
from autodistill.helpers import load_image
from transformers import AutoModelForCausalLM, AutoProcessor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_example(task_prompt, processor, model, image, text_input=None, fast_mode=True):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(DEVICE)
    
    # 优化推理参数以提高速度
    if fast_mode:
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=512,  # 减少从1024到512
            early_stopping=True,  # 启用早停
            do_sample=False,
            num_beams=2,  # 减少从3到1，大幅提升速度
        )
    else:
        # 原始高质量模式
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )
    
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text, task=task_prompt, image_size=(image.width, image.height)
    )
    return parsed_answer

class Florence2(DetectionBaseModel):
    def __init__(self, ontology: CaptionOntology, model_id="microsoft/Florence-2-base"):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, local_files_only=True, device_map=DEVICE
        ).eval()
        self.processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True, local_files_only=True, device_map=DEVICE
        )
        self.ontology = ontology
        
        # 添加性能优化选项
        self.fast_mode = True  # 默认启用快速模式
        
        # 启用模型优化
        if torch.cuda.is_available():
            self.model = torch.compile(self.model, mode="reduce-overhead")

    def set_fast_mode(self, fast_mode: bool):
        """设置是否使用快速推理模式"""
        self.fast_mode = fast_mode

    def predict(self, input: str, confidence: float = 0.5) -> sv.Detections:
        image = load_image(input, return_format="PIL")
        ontology_classes = self.ontology.classes()
        
        # 如果只有一个类别，直接处理
        if len(ontology_classes) == 1:
            result = run_example(
                "<OPEN_VOCABULARY_DETECTION>",
                self.processor,
                self.model,
                image,
                ontology_classes[0],
                fast_mode=self.fast_mode
            )
            results = result["<OPEN_VOCABULARY_DETECTION>"]
        else:
            # 多类别处理
            for i, class_name in enumerate(ontology_classes):
                result = run_example(
                    "<OPEN_VOCABULARY_DETECTION>",
                    self.processor,
                    self.model,
                    image,
                    class_name,
                    fast_mode=self.fast_mode
                )
                if i == 0:
                    results = result["<OPEN_VOCABULARY_DETECTION>"]
                else:
                    if result["<OPEN_VOCABULARY_DETECTION>"]["bboxes"]:
                        results["bboxes"].extend(result["<OPEN_VOCABULARY_DETECTION>"]["bboxes"])
                        results["bboxes_labels"].extend(result["<OPEN_VOCABULARY_DETECTION>"]["bboxes_labels"])

        boxes_and_labels = list(zip(results["bboxes"], results["bboxes_labels"]))
        if len([box for box, label in boxes_and_labels if label in ontology_classes]) == 0:
            return sv.Detections.empty()

        detections = sv.Detections(
            xyxy=np.array([box for box, label in boxes_and_labels if label in ontology_classes]),
            class_id=np.array([ontology_classes.index(label) for box, label in boxes_and_labels if label in ontology_classes]),
            confidence=np.array([1.0 for box, label in boxes_and_labels if label in ontology_classes]),
        )
        return detections[detections.confidence > confidence]