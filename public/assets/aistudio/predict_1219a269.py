"""
python -m via_yoloX.predict
"""

import os
import pickle
import time
import json
from tqdm import tqdm
from pathlib import Path

import cv2
import numpy as np
import torch
from yolox.data.data_augment import ValTransform
from yolox.utils import postprocess

# from inference.main.utils.end_points import GPU_END_POINT
# from ModelInference.main.interfaces.modelwise_inference_interface import ModelWiseInferenceInterface
# from ModelInference.main.modelwise_inference.utils import model_wise_inference_utils
# from ModelInference.main.utils import constants

IDX2LABEL = "idx2label"
CONF_THRE = "conf_thre"
NMS_THRE = "nms_thre"
TEST_SIZE = "test_size"
BBOXES = "bboxes"
LABELS = "label"
SCORES = "scores"
CONFIG_FILE_NAME = "config.json"
CUDA = "cuda"
CPU = "cpu"
VARIANT = "variant"
MODEL_FILE_NAME = "yoloX_S_iter6.pth"
MODEL_FILE_NAME = "yoloX_M_iter4.pth"

class YoloXInference():
    def __init__(self, model_path, images_dir, output_dir):
            self.model_path = model_path
            with open(os.path.join(self.model_path, "config.json"), "r", encoding="utf-8") as f:
                self.additional_data = json.load(f)
            self._device = torch.device(CUDA if torch.cuda.is_available() else CPU)
            self._result = {}
            
            self.images_dir = images_dir
            self.output_dir = output_dir
            
            self.idx2label = self.additional_data[IDX2LABEL]
            self.num_classes = len(self.idx2label)
    
            self.conf_thre = self.additional_data.get(CONF_THRE)
            self.nms_thre = self.additional_data.get(NMS_THRE)
            self.test_size = tuple(self.additional_data.get(TEST_SIZE))
    
            self.preproc = ValTransform(legacy=False)
            self.time = {}
            self.total_time_in_milliseconds = 0.0
            self.load_yolox()
            
    def _infer_yolox_config(self, state_dict):
        """Infer variant and num_classes from YOLOX model state_dict.

        Uses the classification prediction head weight shape:
        head.cls_preds.0.weight  →  [num_classes, int(256*width), 1, 1]

        width → variant mapping (standard YOLOX variants):
        int(256*0.375)=96  → yolox-t
        int(256*0.5)=128   → yolox-s
        int(256*0.75)=192  → yolox-m
        int(256*1.0)=256   → yolox-l
        """
        result = {}
        cls_key = "head.cls_preds.0.weight"
        if cls_key in state_dict:
            weight = state_dict[cls_key]
            result["num_classes"] = weight.shape[0]
            in_ch = weight.shape[1]
            width_channel_to_variant = {96: "yolox-t", 128: "yolox-s", 192: "yolox-m", 256: "yolox-l"}
            result["variant"] = width_channel_to_variant.get(in_ch)
        return result

    def load_yolox(self):
        from yolox.exp import get_exp

        # Load config.json if present; it is optional — missing values are inferred from checkpoint.
        json_path = os.path.join(os.path.dirname(self.model_path), CONFIG_FILE_NAME)
        if os.path.exists(json_path):
            with open(json_path, 'r') as json_file:
                self._additional_data = json.load(json_file)
        else:
            self._additional_data = {}

        # Load checkpoint first so state_dict is available for inference.
        map_loc = CUDA if torch.cuda.is_available() else CPU
        ckpt = torch.load(
            os.path.join(self.model_path, MODEL_FILE_NAME),
            map_location=map_loc,
            weights_only=False
        )

        if isinstance(ckpt, dict):
            state_dict = ckpt.get("model", ckpt)
        elif hasattr(ckpt, "state_dict"):
            state_dict = ckpt.state_dict()
        else:
            raise ValueError(
                f"Unsupported checkpoint type: {type(ckpt)}"
            )

        # Infer variant and num_classes from state_dict when not present in config.json.
        inferred = self._infer_yolox_config(state_dict)

        variant = self._additional_data.get(VARIANT) or inferred.get("variant")
        if variant is None:
            raise ValueError(
                "YOLOX variant not found in config.json and could not be inferred from the checkpoint. "
                "Please provide a config.json with a 'variant' key (e.g. 'yolox-l')."
            )

        # num_classes: prefer idx2label length from config, fall back to inferred count.
        if IDX2LABEL in self._additional_data:
            num_classes = len(self._additional_data[IDX2LABEL])
        else:
            num_classes = inferred.get("num_classes")
            if not num_classes:
                raise ValueError(
                    "num_classes could not be determined. Provide 'idx2label' in config.json."
                )
            # Generate generic labels when idx2label is absent.
            self._additional_data[IDX2LABEL] = {str(i): f"class_{i}" for i in range(num_classes)}

        default_exp_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "utils", "yolox_default_exp.py"
        )
        exp = get_exp(os.path.normpath(default_exp_path), None)
        exp.set_model_scale(variant)

        # config.json values always override the default exp file.
        exp.num_classes = len(self.additional_data[IDX2LABEL])
        if CONF_THRE in self._additional_data:
            exp.test_conf = self._additional_data[CONF_THRE]
        if NMS_THRE in self._additional_data:
            exp.nmsthre = self._additional_data[NMS_THRE]
        if TEST_SIZE in self._additional_data:
            exp.test_size = tuple(self._additional_data[TEST_SIZE])

        model = exp.get_model()
        model.load_state_dict(state_dict)
        model.eval()
        model.additional_data = self._additional_data
        model.exp = exp
        self._model = model
        
    def perform_inference(self, img_path):
        im_read_start_time = time.time() * 1000.0
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        im_time = (time.time() * 1000.0) - im_read_start_time
        print("\nImage Read Time: ", round(im_time,2) , "ms")
        
        starttime = time.time() * 1000.0

        if img is None:
            raise ValueError(f"Unable to read image: {img}")

        # OpenCV loads as BGR, while your existing pipeline expects RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w = img.shape[:2]
        ratio = min(self.test_size[0] / h, self.test_size[1] / w)

        img_tensor, _ = self.preproc(img, None, self.test_size)
        img_tensor = torch.from_numpy(img_tensor).unsqueeze(0).float()

        img_tensor = img_tensor.to(self._device)

        with torch.no_grad():
            raw_output = self._model(img_tensor)
        
        det = postprocess(raw_output, self.num_classes, self.conf_thre, self.nms_thre, class_agnostic=True)[0]
        m_time = (time.time() * 1000.0) - starttime
        print("Model Infer Time: ",  round(m_time,2) , " ms\n")

        updated_boxes = []
        updated_classes = []
        updated_confidence = []

        if det is not None:
            det = det.cpu()
            updated_boxes = (det[:, :4] / ratio).tolist()
            updated_confidence = (det[:, 4] * det[:, 5]).tolist()
            updated_classes = [self.idx2label[str(int(cid))] for cid in det[:, 6].tolist()] # type: ignore

        image_result = {BBOXES: updated_boxes, LABELS: updated_classes,
                        SCORES: updated_confidence}
        
        print(image_result)
        print("Total Inference Time: ", round(m_time + im_time,2), " ms")
        self.time[img_path] = {"image_read_time",im_time,
                               "modelling_time",m_time,
                               "total_time",m_time + im_time}
        self.total_time_in_milliseconds += m_time + im_time
        return image_result
 
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

def collect_images(input_dir: Path) -> list[Path]:
    paths = [
        p for p in sorted(input_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return paths

model_path = "/home/khushal/Downloads/IDX_Data/MODELS/IDX_and_Small_Aadhaar_No_and_VID_Trained_Model_Sept8"

image_dir = "/home/khushal/Downloads/IDX_Data/corrected_labels_jun_2026_filtered_for_training/aadhaar_images_with_VID_information/image_data"
output_dir = "/home/khushal/Downloads/IDX_Data/corrected_labels_jun_2026_filtered_for_training/aadhaar_images_with_VID_information/predictions_iter6__yolox_S_model_sept8"

model_path = "/home/khushal/Downloads/idx_medium_model_run4_oct_5"

image_dir = "/home/khushal/Downloads/IDX_Data/TEST_SET_images_with_small_aadhno_plus_VID_images_plus_New_VoterID_Sept9"
output_dir = "/home/khushal/Downloads/IDX_Data/corrected_labels_jun_2026_filtered_for_training/predictions_iter4__yolox_M_model_oct5"

image_dir = Path(image_dir)


if __name__ == "__main__":
    obj = YoloXInference(model_path, image_dir, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for image_path in tqdm(collect_images(image_dir)):
        output_file = os.path.join(output_dir, os.path.basename(image_path) + ".json")
        if os.path.exists(output_file):
            print(f"Output file already exists for {image_path}. Skipping inference.")
            continue
        result = obj.perform_inference(image_path)
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        
    print("Per Image Breakdown :", obj.time)
    print("Total Time :", round(obj.total_time_in_milliseconds,2), "ms")

"""
python -m src.yolox.predict
"""