cd /d "%~dp0"

call conda activate cyo

echo Starting FastAPI Server...
uvicorn cyo.main:app --host 127.0.0.1 --port 8000 --reload

pause