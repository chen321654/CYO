from fastapi import UploadFile
from pathlib import Path
import os
import shutil

def save_file(file: UploadFile, save_path: str):
    # path = Path(save_dir)
    # path.mkdir(parents=True, exist_ok=True)

    # file_path = os.path.join(save_dir, file.filename)

    with open(save_path, 'wb') as f:
        shutil.copyfileobj(file.file, f)


def delete_file(file_path: str):
    if os.path.exists(file_path):
        os.remove(file_path)
