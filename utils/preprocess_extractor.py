# utils/preprocessor.py
"""
Data preprocessing for landmark sequences
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Tuple, Dict
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    """Preprocess landmark data for BMNet training"""
    
    def __init__(self, input_dim: int = 1404):
        self.input_dim = input_dim
        self.scaler = StandardScaler()
        self.mean = None
        self.std = None
        self.is_fitted = False
    
    def remove_zero_sequences(self, landmarks: np.ndarray, labels: np.ndarray, 
                               threshold: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove videos where most frames have no face detected
        
        Args:
            landmarks: Shape (N, frames, features)
            labels: Shape (N,)
            threshold: Minimum ratio of non-zero frames required
        
        Returns:
            Filtered landmarks and labels
        """
        valid_indices = []
        
        for i in range(len(landmarks)):
            # Check ratio of non-zero frames
            frame_means = np.mean(landmarks[i], axis=1)
            non_zero_ratio = np.sum(frame_means > threshold) / landmarks.shape[1]
            
            # Keep if at least 30% of frames have faces
            if non_zero_ratio >= 0.3:
                valid_indices.append(i)
            else:
                print(f"Removing video {i}: only {non_zero_ratio:.1%} frames have faces")
        
        filtered_landmarks = landmarks[valid_indices]
        filtered_labels = labels[valid_indices]
        
        print(f"Removed {len(landmarks) - len(valid_indices)} videos with insufficient faces")
        print(f"Kept {len(valid_indices)} videos for training")
        
        return filtered_landmarks, filtered_labels
    
    def normalize_sequence(self, landmarks_seq: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        Normalize landmark sequences using standardization
        
        Args:
            landmarks_seq: Shape (N, frames, features) or (frames, features)
            fit: Whether to fit the scaler
        
        Returns:
            Normalized sequence
        """
        # Handle single video
        if landmarks_seq.ndim == 2:
            landmarks_seq = landmarks_seq.reshape(1, *landmarks_seq.shape)
            single_video = True
        else:
            single_video = False
        
        original_shape = landmarks_seq.shape
        
        # Reshape to 2D for scaling
        flat_data = landmarks_seq.reshape(-1, self.input_dim)
        
        # Remove zero rows for fitting (to avoid bias)
        if fit:
            non_zero_mask = np.sum(flat_data, axis=1) > 0.01
            if np.sum(non_zero_mask) > 100:  # Enough samples
                flat_data_for_fit = flat_data[non_zero_mask]
                self.scaler.fit(flat_data_for_fit)
                self.mean = self.scaler.mean_
                self.std = self.scaler.scale_
                self.is_fitted = True
                print(f"✅ Fitted scaler on {np.sum(non_zero_mask)} samples")
            else:
                print("⚠️ Not enough non-zero samples for fitting, using unit scaling")
                self.mean = np.zeros(self.input_dim)
                self.std = np.ones(self.input_dim)
                self.is_fitted = True
        
        # Apply normalization
        if self.is_fitted:
            normalized = self.scaler.transform(flat_data)
        else:
            normalized = flat_data
        
        # Reshape back
        normalized = normalized.reshape(original_shape)
        
        if single_video:
            normalized = normalized[0]
        
        return normalized
    
    def save_scaler(self, save_path: Path):
        """Save the fitted scaler"""
        if not self.is_fitted:
            print("⚠️ Scaler not fitted yet")
            return
        
        data = {
            'scaler': self.scaler,
            'mean': self.mean,
            'std': self.std,
            'input_dim': self.input_dim,
            'is_fitted': self.is_fitted
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"✅ Scaler saved to: {save_path}")
    
    def load_scaler(self, load_path: Path):
        """Load a fitted scaler"""
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
            self.scaler = data['scaler']
            self.mean = data['mean']
            self.std = data['std']
            self.input_dim = data['input_dim']
            self.is_fitted = data['is_fitted']
        print(f"✅ Scaler loaded from: {load_path}")


class TemporalAugmentor:
    """Temporal augmentation for video data"""
    
    @staticmethod
    def add_noise(sequence: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
        """Add Gaussian noise to landmarks"""
        noise = np.random.normal(0, noise_level, sequence.shape)
        return sequence + noise
    
    @staticmethod
    def temporal_shift(sequence: np.ndarray, shift_range: int = 3) -> np.ndarray:
        """Shift sequence temporally"""
        shift = np.random.randint(-shift_range, shift_range)
        if shift > 0:
            return np.roll(sequence, shift, axis=0)
        elif shift < 0:
            return np.roll(sequence, shift, axis=0)
        return sequence
    
    @staticmethod
    def time_reverse(sequence: np.ndarray, prob: float = 0.3) -> np.ndarray:
        """Reverse sequence with probability"""
        if np.random.random() < prob:
            return sequence[::-1]
        return sequence