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

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
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
      setProcessingStatus(`Processing complete: ${response.data.message}`);
      setResults(response.data.results);
      setProcessedVideoUrl(`http://localhost:8000/processed_video/${filename}`);
    } catch (error) {
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
        {processedVideoUrl && results && (
          <section className="results">
            <h2>Processed Video:</h2>
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
            <h3>Commentary:</h3>
            <ul>
              {results.commentary.map((comment, index) => (
                <li key={index}>{comment}</li>
              ))}
            </ul>
          </section>
        )}
      </main>
    </div>
  );
}

export default App;