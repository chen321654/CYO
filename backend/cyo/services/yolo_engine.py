import torch
import torch.nn as nn
import cv2 as cv
from cyo.utils.general import non_max_suppression, scale_boxes
import sys
import onnxruntime as ort
import numpy as np

class YOLOEngine:
    def __init__(self, model_path):
        self.model_path = model_path
        self.session, self.device = self._load_onnx_session()
        self.input_name = self.session.get_inputs()[0].name
        self.names = self._get_model_names()

    def _load_onnx_session(self):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(self.model_path, providers=providers)
        cur_provider = session.get_providers()[0]
        device = 'cuda' if 'CUDA' in cur_provider else 'cpu'
        return session, device
    
    def _get_model_names(self):
        meta = self.session.get_modelmeta().custom_metadata_map
        if 'names' in meta:
            import ast
            return ast.literal_eval(meta['names'])
        return {i: f"class_{i}" for i in range(100)}

    def predict(self, img_path, img_size, conf_thres=0.25, 
                iou_thres=0.45, max_det=1000):
        img0 = cv.imread(img_path)
        img = cv.cvtColor(img0, cv.COLOR_BGR2RGB)
        h0, w0 = img.shape[:2]
        img = cv.resize(img, (img_size, img_size))
        img = img.transpose((2, 0, 1))
        img = np.ascontiguousarray(img).astype(np.float32)
        img /= 255.0
        img = img[None]

        outputs = self.session.run(None, {self.input_name: img})
        pred = outputs[0]
        pred_tensor = torch.from_numpy(pred)

        if len(pred_tensor.shape) == 3 and pred_tensor.shape[2] > pred_tensor.shape[1]:
            pred_tensor = pred_tensor.transpose(1, 2)
        
        results = non_max_suppression(pred_tensor, conf_thres, iou_thres, max_det=max_det)

        detections = []
        for det in results:
            if len(det):
                det[:, :4] = scale_boxes([img_size, img_size], det[:, :4], (h0, w0)).round()
                for *xyxy, conf, cls in det:
                    detections.append({
                        "bbox": [int(x) for x in xyxy],
                        "class_id": int(cls),
                        "class_name": self.names.get(int(cls), f"id_{int(cls)}"),
                        "confidence": float(conf)
                    })
        return detections

        