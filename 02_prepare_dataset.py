# prepare_dataset.py
"""
Main script to prepare FF++ dataset with MediaPipe and C23 quality
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import random
import json
from sklearn.model_selection import train_test_split

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent))

from config import config
from utils.landmark_extractor import MediaPipeLandmarkExtractor
from utils.preprocessor import DataPreprocessor

def setup_directories():
    """Create necessary directories"""
    dirs = [
        config.PROCESSED_DATA_PATH,
        config.PROCESSED_DATA_PATH / 'train',
        config.PROCESSED_DATA_PATH / 'val',
        config.PROCESSED_DATA_PATH / 'test',
        config.PROCESSED_DATA_PATH / 'landmarks_raw',
        config.PROCESSED_DATA_PATH / 'metadata',
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print("✅ Directories created")
    return dirs

def find_videos_c23(dataset_path: Path, num_videos: int = 3000,
                    fake_ratio: float = 2 / 3) -> pd.DataFrame:
    """
    Find videos using C23 quality version (better face detection)
    
    Args:
        dataset_path: Path to FF++ dataset
        num_videos: Total number of videos to use
        fake_ratio: Ratio of fake videos
    
    Returns:
        DataFrame with video paths and labels
    """
    print("\n" + "="*60)
    print("📍 FINDING VIDEOS (C23 QUALITY)")
    print("="*60)
    
    # Quality version: c23 (better than raw)
    quality = config.QUALITY_VERSION
    
    # Real videos path
    real_path = dataset_path / "original"
    if not real_path.exists():
        real_path = dataset_path / "original_sequences" / "youtube" / quality / "videos"
    
    if not real_path.exists():
        print(f"❌ Real videos not found at: {real_path}")
        print("   Trying alternative paths...")
        
        # Try other paths
        alternatives = [
            dataset_path / "original",
            dataset_path / "original_sequences" / "youtube" / "raw" / "videos",
            dataset_path / "original_sequences" / "youtube" / "c40" / "videos",
        ]
        
        for alt in alternatives:
            if alt.exists():
                real_path = alt
                print(f"   ✅ Found at: {real_path}")
                break
    
    # Fake videos path (Deepfakes with c23 quality)
    fake_path = dataset_path / "Deepfakes"
    if not fake_path.exists():
        fake_path = dataset_path / "manipulated_sequences" / "Deepfakes" / quality / "videos"
    
    if not fake_path.exists():
        print(f"   Trying alternative fake paths...")
        alternatives = [
            dataset_path / "Deepfakes",
            dataset_path / "manipulated_sequences" / "Deepfakes" / "raw" / "videos",
            dataset_path / "manipulated_sequences" / "Deepfakes" / "c40" / "videos",
        ]
        
        for alt in alternatives:
            if alt.exists():
                fake_path = alt
                print(f"   ✅ Found fake videos at: {fake_path}")
                break
    
    # Get video lists
    real_videos = sorted(real_path.glob("*.mp4")) if real_path.exists() else []
    fake_videos = sorted(fake_path.glob("*.mp4")) if fake_path.exists() else []
    
    print(f"\n📹 Found:")
    print(f"   Real videos: {len(real_videos)}")
    print(f"   Fake videos: {len(fake_videos)}")
    
    if len(real_videos) == 0 or len(fake_videos) == 0:
        print("\n❌ ERROR: Could not find videos!")
        print(f"   Check your dataset path: {dataset_path}")
        print(f"   Expected structure:")
        print(f"   - {dataset_path}/original_sequences/youtube/{quality}/videos/*.mp4")
        print(f"   - {dataset_path}/manipulated_sequences/Deepfakes/{quality}/videos/*.mp4")
        sys.exit(1)
    
    # Calculate counts
    num_fake = int(round(num_videos * fake_ratio))
    num_real = num_videos - num_fake
    num_real = min(num_real, len(real_videos))
    num_fake = min(num_fake, len(fake_videos))
    
    # Sample videos
    selected_real = random.sample(real_videos, num_real)
    selected_fake = random.sample(fake_videos, num_fake)
    
    # Create DataFrame
    data = []
    for video in selected_real:
        data.append({
            'video_path': str(video),
            'label': 0,
            'type': 'real',
            'forgery_type': 'original',
            'filename': video.name
        })
    
    for video in selected_fake:
        data.append({
            'video_path': str(video),
            'label': 1,
            'type': 'fake',
            'forgery_type': 'Deepfakes',
            'filename': video.name
        })
    
    df = pd.DataFrame(data)
    
    # Shuffle
    df = df.sample(frac=1, random_state=config.RANDOM_SEED).reset_index(drop=True)
    
    print(f"\n📊 Selected Dataset:")
    print(f"   Total: {len(df)} videos")
    print(f"   Real: {len(df[df['label']==0])}")
    print(f"   Fake: {len(df[df['label']==1])}")
    
    return df

def extract_landmarks_with_mediapipe(df: pd.DataFrame, max_frames: int = 30) -> dict:
    """
    Extract landmarks using MediaPipe
    
    Args:
        df: DataFrame with video paths
        max_frames: Maximum frames per video
    
    Returns:
        Dictionary mapping video paths to landmarks
    """
    print("\n" + "="*60)
    print("🎯 EXTRACTING LANDMARKS (MediaPipe - 468 points)")
    print("="*60)
    
    # Initialize MediaPipe extractor
    extractor = MediaPipeLandmarkExtractor(
        static_image_mode=True,
        min_detection_confidence=0.3
    )
    
    # Cache path includes dataset size/frame count so old 1000-video caches are
    # not reused for the full 3000-video dataset.
    cache_path = (
        config.PROCESSED_DATA_PATH
        / 'landmarks_raw'
        / f'landmarks_cache_{config.QUALITY_VERSION}_{len(df)}v_{max_frames}f.npz'
    )
    
    # Check if cache exists
    if cache_path.exists():
        print(f"📦 Loading cached landmarks from: {cache_path}")
        try:
            landmarks_data = np.load(cache_path, allow_pickle=True)
            landmarks_dict = {k: landmarks_data[k] for k in landmarks_data.files}
            print(f"✅ Loaded {len(landmarks_dict)} videos from cache")
            requested_paths = set(df['video_path'].values)
            cached_paths = set(landmarks_dict.keys())
            if requested_paths.issubset(cached_paths):
                return {path: landmarks_dict[path] for path in df['video_path'].values}

            print("   Cache does not match requested dataset. Re-extracting landmarks...")
        except Exception as e:
            print(f"⚠️ Could not load cache: {e}")
            print("   Re-extracting landmarks...")
    
    # Extract landmarks
    video_paths = [Path(p) for p in df['video_path'].values]
    
    landmarks_dict = extractor.extract_batch(
        video_paths, 
        max_frames=max_frames,
        save_path=cache_path
    )
    
    return landmarks_dict

def filter_valid_videos(landmarks_dict: dict, df: pd.DataFrame, 
                        min_valid_frames_ratio: float = 0.3) -> tuple:
    """
    Filter videos with sufficient face detections
    
    Args:
        landmarks_dict: Dictionary of landmarks
        df: Original DataFrame
        min_valid_frames_ratio: Minimum ratio of frames with faces
    
    Returns:
        Filtered landmarks_dict and df
    """
    print("\n" + "="*60)
    print("🔍 FILTERING VALID VIDEOS")
    print("="*60)
    
    valid_paths = []
    valid_frames_count = []
    
    for video_path, landmarks in landmarks_dict.items():
        # Check how many frames have non-zero landmarks
        frame_means = np.mean(landmarks, axis=1)
        valid_frames = np.sum(frame_means > 0.01)
        valid_ratio = valid_frames / landmarks.shape[0]
        
        valid_frames_count.append(valid_ratio)
        
        if valid_ratio >= min_valid_frames_ratio:
            valid_paths.append(video_path)
    
    # Filter DataFrame
    filtered_df = df[df['video_path'].isin(valid_paths)].copy()
    
    # Filter landmarks_dict
    filtered_landmarks = {k: v for k, v in landmarks_dict.items() if k in valid_paths}
    
    # Calculate statistics
    avg_valid_ratio = np.mean(valid_frames_count)
    
    print(f"\n📊 Filtering Results:")
    print(f"   Total videos: {len(landmarks_dict)}")
    print(f"   Valid videos (>{min_valid_frames_ratio:.0%} faces): {len(filtered_landmarks)}")
    print(f"   Removed videos: {len(landmarks_dict) - len(filtered_landmarks)}")
    print(f"   Average face detection rate: {avg_valid_ratio:.1%}")
    
    return filtered_landmarks, filtered_df

def save_data_splits(landmarks_dict: dict, df: pd.DataFrame, preprocessor: DataPreprocessor):
    """
    Save landmarks to train/val/test splits
    """
    print("\n" + "="*60)
    print("💾 SAVING DATA SPLITS")
    print("="*60)
    
    # Split indices
    indices = list(range(len(df)))
    stratify_labels = df['label'] if df['label'].value_counts().min() >= 2 else None
    train_idx, temp_idx = train_test_split(
        indices, 
        test_size=(config.VAL_RATIO + config.TEST_RATIO), 
        random_state=config.RANDOM_SEED, 
        stratify=stratify_labels
    )
    temp_labels = df.iloc[temp_idx]['label']
    temp_stratify = temp_labels if temp_labels.value_counts().min() >= 2 else None
    val_idx, test_idx = train_test_split(
        temp_idx, 
        test_size=(config.TEST_RATIO / (config.VAL_RATIO + config.TEST_RATIO)),
        random_state=config.RANDOM_SEED, 
        stratify=temp_stratify
    )
    
    splits = {
        'train': train_idx,
        'val': val_idx,
        'test': test_idx
    }
    
    split_stats = {}
    
    for split_name, split_indices in splits.items():
        print(f"\n📁 Processing {split_name.upper()} split...")
        
        # Get data for this split
        split_df = df.iloc[split_indices].reset_index(drop=True)
        
        split_landmarks = []
        split_labels = []
        
        for idx, row in split_df.iterrows():
            video_path = row['video_path']
            if video_path in landmarks_dict:
                landmarks = landmarks_dict[video_path]
                split_landmarks.append(landmarks)
                split_labels.append(row['label'])
        
        if len(split_landmarks) == 0:
            print(f"   ⚠️ No valid videos in {split_name} split!")
            continue
        
        split_landmarks = np.array(split_landmarks)
        split_labels = np.array(split_labels)
        
        print(f"   Raw shape: {split_landmarks.shape}")
        
        # Remove videos with too many zeros (only for train)
        if split_name == 'train':
            split_landmarks, split_labels = preprocessor.remove_zero_sequences(
                split_landmarks, split_labels, threshold=0.01
            )
        
        # Normalize
        if split_name == 'train':
            split_landmarks_norm = preprocessor.normalize_sequence(split_landmarks, fit=True)
        else:
            split_landmarks_norm = preprocessor.normalize_sequence(split_landmarks, fit=False)
        
        # Save
        landmarks_save_path = config.PROCESSED_DATA_PATH / split_name / 'landmarks.npy'
        labels_save_path = config.PROCESSED_DATA_PATH / split_name / 'labels.csv'
        
        np.save(landmarks_save_path, split_landmarks_norm)
        split_df.to_csv(labels_save_path, index=False)
        
        print(f"   ✅ Saved: {landmarks_save_path}")
        print(f"      Shape: {split_landmarks_norm.shape}")
        print(f"      Real: {len(split_df[split_df['label']==0])}")
        print(f"      Fake: {len(split_df[split_df['label']==1])}")
        
        split_stats[split_name] = {
            'num_samples': len(split_df),
            'num_real': int((split_df['label'] == 0).sum()),
            'num_fake': int((split_df['label'] == 1).sum()),
            'landmarks_shape': split_landmarks_norm.shape
        }
        
        # Save metadata
        metadata = {
            'split': split_name,
            'num_samples': len(split_df),
            'num_real': int((split_df['label'] == 0).sum()),
            'num_fake': int((split_df['label'] == 1).sum()),
            'input_shape': list(split_landmarks_norm.shape[1:]),
            'quality_version': config.QUALITY_VERSION,
            'num_landmarks': config.NUM_LANDMARKS,
            'max_frames': config.MAX_FRAMES_PER_VIDEO
        }
        
        with open(config.PROCESSED_DATA_PATH / split_name / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
    
    # Save scaler
    preprocessor.save_scaler(config.PROCESSED_DATA_PATH / 'scaler.pkl')
    
    return split_stats

def create_summary_report(df: pd.DataFrame, landmarks_dict: dict, split_stats: dict):
    """Create a detailed summary report"""
    
    print("\n" + "="*60)
    print("📊 DATASET PREPARATION SUMMARY")
    print("="*60)
    
    # Calculate face detection stats
    detection_rates = []
    for landmarks in landmarks_dict.values():
        frame_means = np.mean(landmarks, axis=1)
        valid_ratio = np.sum(frame_means > 0.01) / landmarks.shape[0]
        detection_rates.append(valid_ratio)
    
    report = {
        'total_videos_processed': len(df),
        'videos_with_faces': len(landmarks_dict),
        'face_detection_rate': np.mean(detection_rates),
        'avg_frames_with_faces': np.mean([r * config.MAX_FRAMES_PER_VIDEO for r in detection_rates]),
        'quality_version': config.QUALITY_VERSION,
        'num_landmarks': config.NUM_LANDMARKS,
        'max_frames': config.MAX_FRAMES_PER_VIDEO,
        'input_dim': config.INPUT_DIM,
        'splits': split_stats
    }
    
    print(f"\n📈 Overall Statistics:")
    print(f"   Quality Version: {report['quality_version']}")
    print(f"   Total Videos Processed: {report['total_videos_processed']}")
    print(f"   Videos with Faces Detected: {report['videos_with_faces']}")
    print(f"   Face Detection Rate: {report['face_detection_rate']:.1%}")
    print(f"   Average Frames with Faces: {report['avg_frames_with_faces']:.1f} / {config.MAX_FRAMES_PER_VIDEO}")
    
    print(f"\n📂 Split Statistics:")
    for split_name, stats in split_stats.items():
        print(f"   {split_name.upper()}: {stats['num_samples']} videos")
        print(f"      Real: {stats['num_real']}, Fake: {stats['num_fake']}")
        print(f"      Shape: {stats['landmarks_shape']}")
    
    # Save report
    report_path = config.PROCESSED_DATA_PATH / 'dataset_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✅ Report saved to: {report_path}")
    
    # Show success/failure
    if report['face_detection_rate'] > 0.8:
        print(f"\n🎉 EXCELLENT! Face detection rate is >80%")
        print(f"   Your dataset is ready for BMNet training!")
    elif report['face_detection_rate'] > 0.6:
        print(f"\n👍 GOOD! Face detection rate is >60%")
        print(f"   You can proceed with training")
    else:
        print(f"\n⚠️ WARNING: Face detection rate is low ({report['face_detection_rate']:.1%})")
        print(f"   Suggestions:")
        print(f"   1. Try using 'raw' quality instead of 'c23'")
        print(f"   2. Increase min_detection_confidence to 0.3")
        print(f"   3. Check if your dataset is properly downloaded")
    
    return report

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Prepare FF++ dataset with MediaPipe')
    parser.add_argument('--dataset_path', type=str, 
                       default=str(config.RAW_DATA_PATH),
                       help='Path to FF++ dataset')
    parser.add_argument('--num_videos', type=int, default=1000,
                       help='Total number of videos to use')
    parser.add_argument('--fake_ratio', type=float, default=0.5,
                       help='Ratio of fake videos')
    parser.add_argument('--max_frames', type=int, default=30,
                       help='Maximum frames per video')
    parser.add_argument('--quality', type=str, default='c23',
                       choices=['raw', 'c23', 'c40'],
                       help='Quality version to use')
    
    args = parser.parse_args()
    
    # Update config
    config.QUALITY_VERSION = args.quality
    config.MAX_FRAMES_PER_VIDEO = args.max_frames
    
    print("="*60)
    print("🚀 BMNet DATASET PREPARATION")
    print("="*60)
    print(f"\n⚙️ Configuration:")
    print(f"   Dataset path: {args.dataset_path}")
    print(f"   Quality: {config.QUALITY_VERSION}")
    print(f"   Number of videos: {args.num_videos}")
    print(f"   Fake ratio: {args.fake_ratio}")
    print(f"   Max frames: {config.MAX_FRAMES_PER_VIDEO}")
    print(f"   Landmarks: {config.NUM_LANDMARKS} (MediaPipe)")
    
    # Setup directories
    setup_directories()
    
    # Find videos with C23 quality
    df = find_videos_c23(Path(args.dataset_path), args.num_videos, args.fake_ratio)
    
    # Save original metadata
    df.to_csv(config.PROCESSED_DATA_PATH / 'metadata' / 'full_dataset.csv', index=False)
    
    # Extract landmarks with MediaPipe
    landmarks_dict = extract_landmarks_with_mediapipe(df, args.max_frames)
    
    # Filter valid videos
    landmarks_dict, df_filtered = filter_valid_videos(landmarks_dict, df)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(input_dim=config.INPUT_DIM)
    
    # Save data splits
    split_stats = save_data_splits(landmarks_dict, df_filtered, preprocessor)
    
    # Create report
    report = create_summary_report(df_filtered, landmarks_dict, split_stats)
    
    print("\n" + "="*60)
    print("✅ DATASET PREPARATION COMPLETE!")
    print("="*60)
    print(f"\n📁 Processed data saved to: {config.PROCESSED_DATA_PATH}")
    print("\n🎯 Next Steps:")
    print("   1. Train BMNet model: python train_bmnet.py")
    print("   2. Evaluate on test set: python evaluate.py")
    
    return df_filtered, landmarks_dict, report

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    
    main()
