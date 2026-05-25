# utils/landmark_extractor.py
"""
Facial landmark extraction using MediaPipe (468 landmarks with x, y, z)
"""

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, List, Optional
import warnings

warnings.filterwarnings('ignore')

class MediaPipeLandmarkExtractor:
    """Extract 468 facial landmarks using MediaPipe (with x, y, z coordinates)"""
    
    def __init__(self, static_image_mode=True, min_detection_confidence=0.3):
        """
        Initialize MediaPipe Face Mesh
        
        Args:
            static_image_mode: True detects each sampled frame independently
            min_detection_confidence: Minimum confidence for face detection
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=1,
            refine_landmarks=True,  # Adds iris landmarks
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.num_landmarks = 468
        self.landmark_dim = 3
        self.feature_dim = self.num_landmarks * self.landmark_dim
        
    def extract_landmarks_from_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract 468 landmarks from a single frame
        
        Args:
            frame: BGR image from OpenCV
        
        Returns:
            Array of shape (468, 3) or None if no face detected
        """
        if frame is None:
            return None
        
        # Convert BGR to RGB (MediaPipe expects RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            # Get the first face
            face_landmarks = results.multi_face_landmarks[0]
            
            # Extract coordinates (x, y, z normalized to [0,1])
            coords = []
            for landmark in face_landmarks.landmark:
                coords.extend([landmark.x, landmark.y, landmark.z])
            
            landmarks = np.array(coords, dtype=np.float32)
            if landmarks.size > self.feature_dim:
                landmarks = landmarks[:self.feature_dim]
            elif landmarks.size < self.feature_dim:
                landmarks = np.pad(landmarks, (0, self.feature_dim - landmarks.size))

            return landmarks.astype(np.float32)
        
        return None
    
    def extract_from_video(self, video_path: Path, max_frames: int = 30) -> Tuple[np.ndarray, List[int]]:
        """
        Extract landmarks from entire video
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to extract
        
        Returns:
            landmarks_seq: Array of shape (max_frames, 1404)
            valid_frame_indices: List of frame indices where face was detected
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return np.zeros((max_frames, self.feature_dim), dtype=np.float32), []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Skip videos that are too short
        if total_frames < 10:
            cap.release()
            print(f"Warning: Video too short ({total_frames} frames): {video_path.name}")
            return np.zeros((max_frames, self.feature_dim), dtype=np.float32), []
        
        # Sample frames evenly
        if total_frames > max_frames:
            frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
        else:
            frame_indices = list(range(total_frames))
            # Pad if needed
            if len(frame_indices) < max_frames:
                frame_indices.extend([total_frames - 1] * (max_frames - len(frame_indices)))
        
        landmarks_sequence = []
        valid_frame_indices = []
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                # If frame read fails, add zeros
                landmarks_sequence.append(np.zeros(self.feature_dim, dtype=np.float32))
                continue
            
            # Extract landmarks
            landmarks = self.extract_landmarks_from_frame(frame)
            
            if landmarks is not None:
                landmarks_sequence.append(landmarks)
                valid_frame_indices.append(frame_idx)
            else:
                # No face detected, add zeros
                landmarks_sequence.append(np.zeros(self.feature_dim, dtype=np.float32))
        
        cap.release()
        
        # Ensure we have exactly max_frames
        while len(landmarks_sequence) < max_frames:
            landmarks_sequence.append(np.zeros(self.feature_dim, dtype=np.float32))
        
        return np.stack(landmarks_sequence[:max_frames]).astype(np.float32), valid_frame_indices
    
    def extract_batch(self, video_paths: List[Path], max_frames: int = 30, 
                     save_path: Optional[Path] = None) -> dict:
        """
        Extract landmarks for multiple videos
        
        Args:
            video_paths: List of video paths
            max_frames: Maximum frames per video
            save_path: Optional path to save extracted landmarks
        
        Returns:
            Dictionary with video paths as keys and landmarks as values
        """
        landmarks_data = {}
        failed_videos = []
        
        for video_path in tqdm(video_paths, desc="Extracting landmarks (MediaPipe)"):
            landmarks_seq, valid_frames = self.extract_from_video(video_path, max_frames)
            
            # Check if we have any valid landmarks
            if len(valid_frames) == 0:
                failed_videos.append(str(video_path))
                print(f"⚠️ No face detected in: {video_path.name}")
            
            landmarks_data[str(video_path)] = landmarks_seq
        
        print(f"\n✅ Processed {len(landmarks_data)} videos")
        print(f"⚠️ No face detected in {len(failed_videos)} videos")
        
        if save_path:
            # Save as compressed numpy archive
            np.savez_compressed(save_path, **landmarks_data)
            print(f"💾 Saved landmarks to: {save_path}")
        
        return landmarks_data

    def visualize_landmarks(self, frame: np.ndarray, save_path: Optional[Path] = None):
        """Visualize landmarks on a frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            annotated_frame = frame.copy()
            for face_landmarks in results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    face_landmarks,
                    self.mp_face_mesh.FACEMESH_CONTOURS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                    self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1)
                )
            
            if save_path:
                cv2.imwrite(str(save_path), annotated_frame)
            
            return annotated_frame
        
        return frame


class DlibLandmarkExtractor:
    """Fallback extractor using Dlib (68 landmarks) - in case MediaPipe fails"""
    
    def __init__(self, predictor_path='shape_predictor_68_face_landmarks.dat'):
        try:
            import dlib
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(predictor_path) if Path(predictor_path).exists() else None
            self.num_landmarks = 68
            self.available = self.predictor is not None
        except ImportError:
            self.available = False
            print("Dlib not available")
    
    def extract_from_video(self, video_path, max_frames=30):
        """Extract 68 landmarks - simplified version"""
        # This is a fallback, not fully implemented
        return np.zeros((max_frames, 136)), []
