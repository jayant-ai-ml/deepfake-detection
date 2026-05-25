# config.py
"""
Configuration file for BMNet dataset preparation with MediaPipe
"""

from pathlib import Path
import os

class Config:
    # Paths
    BASE_PATH = Path(__file__).resolve().parent
    RAW_DATA_PATH = BASE_PATH / "data"
    PROCESSED_DATA_PATH = BASE_PATH / "data" / "processed"
    
    # Dataset splits
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # Video processing
    MAX_FRAMES_PER_VIDEO = 30
    TARGET_FPS = 30
    MIN_FRAMES = 10
    
    # MediaPipe Landmarks (468 points with x, y, z)
    NUM_LANDMARKS = 468
    LANDMARK_DIM = 3  # x, y, z coordinates
    INPUT_DIM = 1404  # 468 * 3
    
    # Quality version - USE C23 (better than RAW for face detection)
    QUALITY_VERSION = "c23"  # Options: 'raw', 'c23', 'c40'
    
    # Forgery types to use
    FORGERY_TYPES = ['Deepfakes']  # Can add: 'FaceSwap', 'Face2Face', 'NeuralTextures'
    
    # Random seed
    RANDOM_SEED = 42

config = Config()
