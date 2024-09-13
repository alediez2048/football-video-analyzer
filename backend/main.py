from fastapi import FastAPI, File, UploadFile, WebSocket
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

# Store active WebSocket connections
active_connections = {}

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
        return {"error": "File not found"}
    
    try:
        # Remove the asyncio.to_thread wrapper, as process_video is already async
        results = await asyncio.wait_for(process_video(file_path), timeout=900)  # 15 minutes timeout
        return results
    except asyncio.TimeoutError:
        logger.error("Video processing timed out")
        return {"error": "Video processing timed out"}
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}", exc_info=True)
        return {"error": f"Error processing video: {str(e)}"}

@app.get("/processed_video/{filename}")
async def get_processed_video(filename: str):
    video_path = f"uploads/{filename}"
    return FileResponse(video_path)

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    active_connections[client_id] = websocket
    try:
        while True:
            data = await websocket.receive_text()
            if data == "close":
                break
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        del active_connections[client_id]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React app's address
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Function to send progress updates
async def send_progress_update(client_id: str, progress: int):
    if client_id in active_connections:
        await active_connections[client_id].send_json({"progress": progress})

# Make send_progress_update available to other modules
app.send_progress_update = send_progress_update

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)