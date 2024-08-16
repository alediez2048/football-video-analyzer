from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import shutil
import os
from video_processor import process_video
from fastapi.responses import FileResponse
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

if not os.path.exists("uploads"):
    os.makedirs("uploads")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

@app.get("/")
async def root():
    return {"message": "Welcome to the Football Video Analyzer API"}

@app.post("/upload_video/")
async def upload_video(file: UploadFile = File(...)):
    try:
        with open(f"uploads/{file.filename}", "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"filename": file.filename, "message": "Video uploaded successfully"}
    except Exception as e:
        return {"message": f"There was an error uploading the file: {str(e)}"}

@app.get("/process_video/{filename}")
async def process_video_endpoint(filename: str):
    file_path = f"uploads/{filename}"
    if not os.path.exists(file_path):
        return {"message": "File not found"}
    
    try:
        results = await asyncio.wait_for(asyncio.to_thread(process_video, file_path), timeout=900)  # 5 minutes timeout
    except asyncio.TimeoutError:
        logger.error("Video processing timed out")
        return {"message": "Video processing timed out"}
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}", exc_info=True)
        return {"message": f"Error processing video: {str(e)}"}
    
    return {"message": f"Processed video: {filename}", "results": results}

@app.get("/processed_video/{filename}")
async def get_processed_video(filename: str):
    video_path = f"uploads/{filename}"
    return FileResponse(video_path)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React app's address
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)