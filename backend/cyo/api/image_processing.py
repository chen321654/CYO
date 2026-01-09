from fastapi import APIRouter, UploadFile, File, Form, Request
import shutil 
from cyo.services.chaos_algorithm import encryption, decryption
from cyo.utils.file_process import save_file, delete_file
import uuid
from pathlib import Path

router = APIRouter()

@router.post("/api/predict")
async def target_detect(request: Request, file: UploadFile = File(...)):
    temp_dir = "static/temp"
    path = Path(temp_dir)
    path.mkdir(parents=True, exist_ok=True)
    temp_file_path = f"{temp_dir}/{uuid.uuid4()}_{file.filename}"

    try:
        save_file(file, temp_file_path)

        model = request.app.state.yolo_warship
        detections = model.predict(img_path=temp_file_path, img_size=640)
        # TODO: draw predict frame and store image
        return {
            "detect": detections
        }
    finally:
        delete_file(temp_file_path)


@router.post("/encrypt")
async def test():
    return {"message": "你好！"}

