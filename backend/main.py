from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import shutil
import os
from video_processor import process_video
from fastapi.responses import FileResponse

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
    print(f"Received request to process video: {filename}")
    file_path = f"uploads/{filename}"
    print(f"Checking for file at path: {file_path}")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return {"message": "File not found"}
    
    print("File found, starting processing...")
    results = process_video(file_path)
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