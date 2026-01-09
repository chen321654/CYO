from fastapi import FastAPI
from cyo.api import image_processing
from cyo.services.yolo_engine import YOLOEngine
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.yolo_warship = YOLOEngine(model_path="models/HRSC2016.onnx")
    yield
    del app.state.yolo_warship

app = FastAPI(
    title="CYO",
    description="基于改进YOLOv9的敏感区域图像加密算法",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(image_processing.router, prefix="", tags=["图像处理"])

