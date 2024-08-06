import React, { useRef, useEffect, useState } from 'react';

const ProcessedVideo = ({ videoUrl, detections }) => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [videoDimensions, setVideoDimensions] = useState({ width: 0, height: 0 });

  useEffect(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    const updateCanvasSize = () => {
      const videoRect = video.getBoundingClientRect();
      canvas.width = videoRect.width;
      canvas.height = videoRect.height;
      setVideoDimensions({ width: videoRect.width, height: videoRect.height });
    };

    const drawDetections = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const currentTime = Math.floor(video.currentTime * 24); // Assuming 24 fps
      const currentFrame = detections.find(d => d.frame === currentTime);
      
      if (currentFrame && currentFrame.detections) {
        const scaleX = canvas.width / video.videoWidth;
        const scaleY = canvas.height / video.videoHeight;

        currentFrame.detections.forEach(detection => {
          const [x1, y1, x2, y2] = detection.box;
          const centerX = ((x1 + x2) / 2) * scaleX;
          const bottomY = y2 * scaleY;
          const radius = 5; // Adjust this value to change circle size

          ctx.beginPath();
          ctx.arc(centerX, bottomY, radius, 0, 2 * Math.PI);
          ctx.fillStyle = detection.type === 'sports ball' ? 'red' : 'green';
          ctx.fill();

          ctx.fillStyle = 'white';
          ctx.font = '12px Arial';
          ctx.textAlign = 'center';
          ctx.fillText(
            `${detection.type} ${Math.round(detection.confidence * 100)}%`, 
            centerX, 
            bottomY > canvas.height - 20 ? bottomY - 10 : bottomY + 20
          );
        });
      }

      requestAnimationFrame(drawDetections);
    };

    const handlePlay = () => {
      updateCanvasSize();
      drawDetections();
    };

    video.addEventListener('play', handlePlay);
    window.addEventListener('resize', updateCanvasSize);

    return () => {
      video.removeEventListener('play', handlePlay);
      window.removeEventListener('resize', updateCanvasSize);
    };
  }, [videoUrl, detections]);

  return (
    <div style={{ position: 'relative', width: 'fit-content' }}>
      <video 
        ref={videoRef} 
        src={videoUrl} 
        controls 
        style={{ maxWidth: '100%', display: 'block' }} 
      />
      <canvas 
        ref={canvasRef} 
        style={{ 
          position: 'absolute', 
          top: 0, 
          left: 0, 
          width: `${videoDimensions.width}px`, 
          height: `${videoDimensions.height}px` 
        }} 
      />
    </div>
  );
};

export default ProcessedVideo;