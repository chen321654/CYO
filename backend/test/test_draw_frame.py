import pytest
from cyo.constants import TARGETDETECTION, WARSHIP_MODEL_PATH, CARID_MODEL_PATH, FACE_MODEL_PATH
from cyo.utils.draw_frame import draw_frame
from cyo.services.yolo_engine import YOLOEngine

@pytest.mark.parametrize("img_path, category, model_path", [
    ("static/100001270.jpg", TARGETDETECTION, WARSHIP_MODEL_PATH),
    ("static/36.jpg", TARGETDETECTION, CARID_MODEL_PATH),
    ("static/000018.jpg", TARGETDETECTION, FACE_MODEL_PATH)
])
def test_draw_frame(img_path, category, model_path):
    warship_engine = YOLOEngine(model_path)

    detections = warship_engine.predict(img_path=img_path, img_size=640)

    draw_frame(img_path, detections, category)

