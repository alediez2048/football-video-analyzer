import React, { useRef, useEffect } from 'react';

const ProcessedVideo = ({ videoUrl, detections }) => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  useEffect(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    const drawDetections = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      if (!detections || detections.length === 0) {
        return;  // Exit if there are no detections
      }

      const currentTime = Math.floor(video.currentTime * 24); // Assuming 24 fps
      const currentFrame = detections.find(d => d.frame === currentTime);
      
      if (currentFrame && currentFrame.detections) {
        const scaleX = canvas.width / video.videoWidth;
        const scaleY = canvas.height / video.videoHeight;

        currentFrame.detections.forEach(detection => {
          const [x1, y1, x2, y2] = detection.box;
          const width = (x2 - x1) * scaleX;
          const height = (y2 - y1) * scaleY;

          // Draw bounding box
          ctx.strokeStyle = detection.type === 'sports ball' ? 'red' : 'green';
          ctx.lineWidth = 2;
          ctx.strokeRect(x1 * scaleX, y1 * scaleY, width, height);

          // Draw label
          ctx.fillStyle = 'white';
          ctx.font = '12px Arial';
          ctx.fillText(
            `${detection.type} ${detection.team !== undefined ? detection.team : ''}`, 
            x1 * scaleX, 
            y1 * scaleY > 20 ? y1 * scaleY - 5 : y1 * scaleY + 20
          );
        });
      }

      requestAnimationFrame(drawDetections);
    };

    const handlePlay = () => {
      canvas.width = video.clientWidth;
      canvas.height = video.clientHeight;
      drawDetections();
    };

    video.addEventListener('play', handlePlay);

    return () => {
      video.removeEventListener('play', handlePlay);
    };
  }, [videoUrl, detections]);

  return (
    <div style={{ position: 'relative' }}>
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
          pointerEvents: 'none' 
        }} 
      />
    </div>
  );
};

export default ProcessedVideo;