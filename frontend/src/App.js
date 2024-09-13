// frontend/src/App.js
import React, { useState } from 'react';
import axios from 'axios';
import ProcessedVideo from './ProcessedVideo';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState('');
  const [processingStatus, setProcessingStatus] = useState('');
  const [results, setResults] = useState(null);
  const [processedVideoUrl, setProcessedVideoUrl] = useState(null);
  const [videoDuration, setVideoDuration] = useState(null);

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
    const video = document.createElement('video');
    video.preload = 'metadata';
    video.onloadedmetadata = function() {
      window.URL.revokeObjectURL(video.src);
      setVideoDuration(video.duration);
    }
    video.src = URL.createObjectURL(event.target.files[0]);
  };

  const handleUpload = async () => {
    if (!file) {
      setUploadStatus('Please select a file first');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
      setUploadStatus('Uploading...');
      const response = await axios.post('http://localhost:8000/upload_video/', formData);
      setUploadStatus(`Upload successful: ${response.data.filename}`);
      processVideo(response.data.filename);
    } catch (error) {
      setUploadStatus(`Upload failed: ${error.message}`);
    }
  };

  const processVideo = async (filename) => {
    try {
      setProcessingStatus('Processing video...');
      const response = await axios.get(`http://localhost:8000/process_video/${filename}`);
      console.log('Response from backend:', response.data);  // Keep this line for debugging
      if (response.data.error) {
        setProcessingStatus(`Processing error: ${response.data.error}`);
      } else {
        const totalFrames = response.data.total_frames || 'unknown';
        setProcessingStatus(`Processing complete: ${totalFrames} frames processed`);
      }
      setResults(response.data);
      setProcessedVideoUrl(`http://localhost:8000/processed_video/${filename}`);
    } catch (error) {
      console.error('Error processing video:', error);  // Keep this line for debugging
      setProcessingStatus(`Processing failed: ${error.message}`);
      setResults(null);
    }
  };

  return (
    <div className="App">
      <header className="header">
        <h1>⚽ Football Video Analyzer ⚽</h1>
      </header>
      <main>
        <section className="upload-section">
          <input 
            type="file" 
            onChange={handleFileChange} 
            className="file-input" 
            id="file-input"
          />
          <label htmlFor="file-input" className="file-label">
            Choose File
          </label>
          <button onClick={handleUpload} className="upload-btn">
            Upload and Process
          </button>
          <p className="status">{uploadStatus}</p>
          <p className="status">{processingStatus}</p>
        </section>
        {results && (
          <section className="results">
            <h2>Processed Video:</h2>
            {videoDuration && <p>Video Duration: {videoDuration.toFixed(2)} seconds</p>}
            <ProcessedVideo videoUrl={processedVideoUrl} detections={results.detections} />
            <h3>Events:</h3>
            {results.events && results.events.length > 0 ? (
              <ul>
                {results.events.map((event, index) => (
                  <li key={index}>{event.type} at frame {event.frame}</li>
                ))}
              </ul>
            ) : (
              <p>No events detected</p>
            )}
            {results.commentary && results.commentary.length > 0 && (
              <>
                <h3>Commentary:</h3>
                <ul>
                  {results.commentary.map((comment, index) => (
                    <li key={index}>{comment}</li>
                  ))}
                </ul>
              </>
            )}
            {results.statistics && (
              <section className="statistics">
                <h3>Statistics:</h3>
                {results.statistics.possession && (
                  <p>Possession: Team 0 - {results.statistics.possession[0].toFixed(2)}%, Team 1 - {results.statistics.possession[1].toFixed(2)}%</p>
                )}
                {results.statistics.passes && (
                  <p>Passes: Team 0 - {results.statistics.passes[0]}, Team 1 - {results.statistics.passes[1]}</p>
                )}
                {results.statistics.shots && (
                  <p>Shots: Team 0 - {results.statistics.shots[0]}, Team 1 - {results.statistics.shots[1]}</p>
                )}
              </section>
            )}
          </section>
        )}
      </main>
    </div>
  );
}

export default App;