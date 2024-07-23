from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from video_processor import process_video

app = FastAPI()

# Keep your existing CORS middleware setup and other routes

@app.get("/")
async def root():
    return {"message": "Welcome to the Football Video Analyzer API"}

# Keep your existing upload_video and process_video endpoints