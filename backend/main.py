from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from video_processor import process_video

app = FastAPI()

# ... (keep the existing CORS middleware setup)

@app.post("/upload_video/")
async def upload_video(file: UploadFile = File(...)):
    try:
        os.makedirs("uploads", exist_ok=True)
        file_path = f"uploads/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"filename": file.filename, "message": "Video uploaded successfully"}
    except Exception as e:
        return {"message": f"There was an error uploading the file: {str(e)}"}

@app.get("/process_video/{filename}")
async def process_video_endpoint(filename: str):
    file_path = f"uploads/{filename}"
    if not os.path.exists(file_path):
        return {"message": "File not found"}
    
    results = process_video(file_path)
    return {"message": f"Processed video: {filename}", "results": results}

# ... (keep the existing __main__ block)