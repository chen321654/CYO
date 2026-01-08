from fastapi import FastAPI
from cyo.api import image_processing

app = FastAPI(
    title="CYO",
    description="基于改进YOLOv9的敏感区域图像加密算法",
    version="1.0.0"
)

app.include_router(image_processing.router, prefix="", tags=["图像处理"])