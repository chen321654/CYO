from fastapi import APIRouter, UploadFile, File, Form
import shutil 
from cyo.services.chaos_algorithm import encryption, decryption

router = APIRouter()

@router.post("/encrypt")
async def test():
    return {"message": "你好！"}

