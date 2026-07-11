import cv2
import numpy as np
import os
import shutil
import logging

logger = logging.getLogger(__name__)

class VideoPipeline:
    def __init__(self, upload_dir="data/videos", clips_dir="data/clips"):
        self.upload_dir = upload_dir
        self.clips_dir = clips_dir
        
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.clips_dir, exist_ok=True)

    def process_video(self, video_path, video_id):
        """
        The Main Pipeline:
        1. Decode Video
        2. Generator: Yield Temporal Units (Clips)
        3. Motion Encoder: Compute Optical Flow
        4. Gate: Return metadata only for valid events
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Pipeline Parameters (As defined in Design Doc)
        CLIP_LENGTH = 16  # frames
        STRIDE = 8        # frames (50% overlap)
        
        # Create folder for this video's evidence
        video_output_dir = os.path.join(self.clips_dir, video_id)
        if os.path.exists(video_output_dir):
            shutil.rmtree(video_output_dir)
        os.makedirs(video_output_dir)

        valid_events = []
        
        # Buffer to hold small frames for optical flow and CLIP
        frames_buffer = []
        
        frame_idx = 0
        logger.info(f"Starting frame extraction loop for {video_id} - Total Frames: {total_frames}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx > 0 and frame_idx % 100 == 0:
                logger.info(f"Processed {frame_idx}/{total_frames} frames for {video_id}")
            
            # Resize for performance and memory
            # Downscale to 360p to balance flow speed and visual quality for CLIP
            h, w = frame.shape[:2]
            scale = 360 / h
            small_frame = cv2.resize(frame, (int(w * scale), 360), interpolation=cv2.INTER_NEAREST)
            
            frames_buffer.append(small_frame)
            
            # Maintain sliding window buffer
            if len(frames_buffer) > CLIP_LENGTH:
                frames_buffer.pop(0)
            
            # Process window every 'STRIDE' frames, once buffer is full
            if len(frames_buffer) == CLIP_LENGTH and frame_idx % STRIDE == 0:
                
                # --- MOTION ENCODER (Physics Layer) ---
                motion_score = self._compute_optical_flow_energy(frames_buffer)
                
                # --- ABSTENTION GATE (The "Control Rule") ---
                # If physics says "nothing happened", we ABSTAIN.
                # Threshold can be tuned. 0.5 is a robust baseline for dense flow.
                if motion_score > 0.5:
                    
                    # Save the "Key Frame" (Center of the clip)
                    # This is what CLIP will "see", but it's only allowed to see it
                    # because the Motion Encoder verified it.
                    timestamp = (frame_idx - (CLIP_LENGTH // 2)) / fps
                    
                    # Save the small frame to save disk space and drastically improve speed
                    key_frame_path = os.path.join(video_output_dir, f"{frame_idx}.jpg")
                    cv2.imwrite(key_frame_path, frames_buffer[CLIP_LENGTH // 2])
                    
                    valid_events.append({
                        "video_id": video_id,
                        "timestamp": timestamp,
                        "frame_path": key_frame_path,
                        "display_time": self._format_timestamp(timestamp),
                        "event_probability": motion_score # The evidence confidence
                    })
            
            frame_idx += 1
            
        cap.release()
        
        # Subsample to a maximum of 15 frames to avoid UI timeouts during heavy processing
        if len(valid_events) > 15:
            step = len(valid_events) / 15.0
            valid_events = [valid_events[int(i * step)] for i in range(15)]
            
        return valid_events

    def _compute_optical_flow_energy(self, frames):
        """
        Calculates the magnitude of motion between frames in the clip.
        Uses Dense Optical Flow (Farneback).
        """
        motion_energy = 0
        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        
        # We sample a few frames within the clip to estimate total motion
        # checking every frame is too slow for CPU, checking stride is a good compromise
        for i in range(1, len(frames), 4):
            curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            
            # Compute Dense Flow
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None, 
                pyr_scale=0.5, levels=3, winsize=15, 
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            
            # Calculate magnitude (speed of movement)
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            # Mean magnitude of the entire screen
            motion_energy += np.mean(magnitude)
            prev_gray = curr_gray
            
        return motion_energy

    def _format_timestamp(self, seconds):
        m, s = divmod(seconds, 60)
        return f"{int(m):02d}:{int(s):02d}"