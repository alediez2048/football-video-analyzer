import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [file, setFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState('');
  const [processingStatus, setProcessingStatus] = useState('');

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleUpload = async () => {
    const formData = new FormData();
    formData.append('file', file);

    try {
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
    } catch (error) {
      setProcessingStatus(`Processing failed: ${error.message}`);
    }
  };

  return (
    <div className="App">
      <h1>Football Video Analyzer</h1>
      <input type="file" onChange={handleFileChange} />
      <button onClick={handleUpload}>Upload and Process</button>
      <p>{uploadStatus}</p>
      <p>{processingStatus}</p>
    </div>
  );
}

export default App;