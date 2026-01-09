from cyo.services.yolo_engine import YOLOEngine
import pytest
import logging

@pytest.mark.parametrize("model_path, img_path", [
    ("models/HRSC2016.onnx", "static/100001270.jpg"),
    ("models/CelebA.onnx", "static/000018.jpg"),
    ("models/CCPD.onnx", "static/36.jpg"),
])
def test_yolo(model_path, img_path):
    yolo_engine = YOLOEngine(model_path=model_path)

    detections = yolo_engine.predict(img_path=img_path, img_size=640)

    logging.info(detections)
    # print(detections)
