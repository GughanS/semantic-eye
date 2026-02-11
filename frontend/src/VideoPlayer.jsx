import React, { forwardRef } from 'react';
import { Upload } from 'lucide-react';

const VideoPlayer = forwardRef(({ videoUrl }, ref) => {
  if (!videoUrl) {
    return (
      <div className="video-player-wrapper">
        <div className="video-placeholder">
          <Upload />
          <p>Upload a video to initialize the Eye</p>
        </div>
      </div>
    );
  }

  return (
    <div className="video-player-wrapper">
      <video 
        ref={ref}
        src={videoUrl} 
        controls 
        className="video-element"
      />
    </div>
  );
});

VideoPlayer.displayName = 'VideoPlayer';
export default VideoPlayer;