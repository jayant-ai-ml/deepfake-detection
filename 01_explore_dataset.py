"""
FF++ Dataset Explorer for 1000 Videos (500 Real + 500 Fake)
Author: Project Implementation
"""

import os
import sys
import cv2
import glob
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# For landmark detection
try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False
    print("Warning: dlib not installed. Install with: pip install dlib")

# For face detection
try:
    import mediapipe as mp
    MP_AVAILABLE = hasattr(mp, "solutions")
except ImportError:
    MP_AVAILABLE = False
    print("Warning: mediapipe not installed. Install with: pip install mediapipe")

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

class FFPPDatasetExplorer:
    def __init__(self, dataset_path, num_videos=1000, fake_ratio=0.5):
        """
        Initialize the dataset explorer
        
        Args:
            dataset_path: Path to FaceForensics++ dataset
            num_videos: Total number of videos to use (default: 1000)
            fake_ratio: Ratio of fake videos (default: 0.5 = 500 fake, 500 real)
        """
        self.dataset_path = Path(dataset_path)
        self.num_videos = num_videos
        self.num_real = int(num_videos * (1 - fake_ratio))
        self.num_fake = int(num_videos * fake_ratio)
        
        self.real_videos = []
        self.fake_videos = []
        self.selected_real = []
        self.selected_fake = []
        
        # Initialize face detectors
        self.init_face_detectors()
        
    def init_face_detectors(self):
        """Initialize face detection models"""
        if DLIB_AVAILABLE:
            self.dlib_detector = dlib.get_frontal_face_detector()
            self.dlib_predictor = dlib.shape_predictor(
                'shape_predictor_68_face_landmarks.dat'  # Download if needed
            ) if os.path.exists('shape_predictor_68_face_landmarks.dat') else None
        
        if MP_AVAILABLE:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                min_detection_confidence=0.5
            )
    
    def scan_dataset(self):
        """Scan the dataset directory to find all videos"""
        print("\n" + "="*60)
        print("SCANNING FF++ DATASET")
        print("="*60)
        
        # Find real videos
        real_path = self.dataset_path / "original"
        if real_path.exists():
            self.real_videos = list(real_path.glob("*.mp4"))
            print(f"✅ Found {len(self.real_videos)} real videos")
        else:
            # Try alternative path
            alt_path = self.dataset_path / "original_sequences" / "youtube" / "raw" / "videos"
            if alt_path.exists():
                self.real_videos = list(alt_path.glob("*.mp4"))
                print(f"✅ Found {len(self.real_videos)} real videos (c23 quality)")
            else:
                print(f"❌ Real videos not found at: {real_path}")
                print(f"   Please check your dataset structure")
        
        # Find fake videos (Deepfakes manipulation type)
        fake_path = self.dataset_path / "Deepfakes"
        if not fake_path.exists():
            fake_path = self.dataset_path / "manipulated_sequences" / "Deepfakes" / "raw" / "videos"
        
        if fake_path.exists():
            self.fake_videos = list(fake_path.glob("*.mp4"))
            print(f"✅ Found {len(self.fake_videos)} fake videos (Deepfakes)")
        else:
            print(f"❌ Fake videos not found at: {fake_path}")
        
        # Also check other manipulation types
        other_forgeries = ["FaceSwap", "Face2Face", "NeuralTextures"]
        for forgery in other_forgeries:
            forged_path = self.dataset_path / "manipulated_sequences" / forgery / "raw" / "videos"
            if forged_path.exists():
                other_videos = list(forged_path.glob("*.mp4"))
                print(f"   Found {len(other_videos)} {forgery} videos")
        
        return len(self.real_videos), len(self.fake_videos)
    
    def select_balanced_subset(self):
        """Select balanced subset of real and fake videos"""
        print("\n" + "="*60)
        print("SELECTING BALANCED SUBSET")
        print("="*60)
        
        # Select real videos
        if len(self.real_videos) >= self.num_real:
            self.selected_real = random.sample(self.real_videos, self.num_real)
            print(f"✅ Selected {len(self.selected_real)} real videos")
        else:
            self.selected_real = self.real_videos.copy()
            print(f"⚠️ Only {len(self.real_videos)} real videos available")
        
        # Select fake videos
        if len(self.fake_videos) >= self.num_fake:
            self.selected_fake = random.sample(self.fake_videos, self.num_fake)
            print(f"✅ Selected {len(self.selected_fake)} fake videos")
        else:
            self.selected_fake = self.fake_videos.copy()
            print(f"⚠️ Only {len(self.fake_videos)} fake videos available")
        
        # Create final dataset list
        self.dataset = []
        for video in self.selected_real:
            self.dataset.append({'path': str(video), 'label': 0, 'type': 'real'})
        for video in self.selected_fake:
            self.dataset.append({'path': str(video), 'label': 1, 'type': 'fake'})
        
        # Shuffle
        random.shuffle(self.dataset)
        
        print(f"\n📊 Final Dataset: {len(self.dataset)} videos total")
        print(f"   - Real: {len(self.selected_real)}")
        print(f"   - Fake: {len(self.selected_fake)}")
        
        return pd.DataFrame(self.dataset)
    
    def extract_video_info(self, video_path, num_frames=10):
        """Extract basic information from a video"""
        cap = cv2.VideoCapture(video_path)
        
        info = {
            'path': video_path,
            'filename': Path(video_path).name,
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration_seconds': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        }
        
        # Extract sample frames
        frames = []
        frame_indices = np.linspace(0, info['total_frames'] - 1, num_frames, dtype=int)
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        
        cap.release()
        info['sample_frames'] = frames
        
        return info
    
    def detect_faces_in_frame(self, frame):
        """Detect faces in a frame using multiple methods"""
        face_info = {
            'num_faces': 0,
            'face_bboxes': [],
            'landmarks': None
        }
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Dlib
        if DLIB_AVAILABLE:
            faces = self.dlib_detector(gray)
            face_info['num_faces'] = len(faces)
            for face in faces:
                bbox = [face.left(), face.top(), face.right(), face.bottom()]
                face_info['face_bboxes'].append(bbox)
                
                # Get landmarks if predictor is available
                if self.dlib_predictor:
                    landmarks = self.dlib_predictor(gray, face)
                    landmarks_coords = []
                    for i in range(68):
                        x = landmarks.part(i).x
                        y = landmarks.part(i).y
                        landmarks_coords.append([x, y])
                    face_info['landmarks'] = np.array(landmarks_coords)
        
        return face_info
    
    def analyze_landmarks_over_sequence(self, video_path, num_frames_to_analyze=30):
        """Analyze facial landmarks across a sequence of frames"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample frames evenly
        frame_indices = np.linspace(0, total_frames - 1, num_frames_to_analyze, dtype=int)
        
        landmarks_sequence = []
        frame_faces = []
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret and DLIB_AVAILABLE:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.dlib_detector(gray)
                
                if len(faces) > 0 and self.dlib_predictor:
                    # Get landmarks for first face
                    landmarks = self.dlib_predictor(gray, faces[0])
                    coords = []
                    for i in range(68):
                        coords.append([landmarks.part(i).x, landmarks.part(i).y])
                    landmarks_sequence.append(np.array(coords))
                    frame_faces.append(len(faces))
                else:
                    landmarks_sequence.append(None)
                    frame_faces.append(0)
            else:
                landmarks_sequence.append(None)
                frame_faces.append(0)
        
        cap.release()
        return landmarks_sequence, frame_faces
    
    def compute_statistics(self, df):
        """Compute comprehensive statistics about the dataset"""
        print("\n" + "="*60)
        print("DATASET STATISTICS")
        print("="*60)
        
        stats = {
            'total_videos': len(df),
            'real_videos': len(df[df['type'] == 'real']),
            'fake_videos': len(df[df['type'] == 'fake']),
            'video_stats': {
                'fps': [],
                'duration': [],
                'resolution': []
            }
        }
        
        # Analyze each video
        print("\n📹 Analyzing video properties...")
        video_infos = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing videos"):
            cap = cv2.VideoCapture(row['path'])
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps > 0 else 0
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                stats['video_stats']['fps'].append(fps)
                stats['video_stats']['duration'].append(duration)
                
                video_infos.append({
                    'filename': Path(row['path']).name,
                    'type': row['type'],
                    'fps': fps,
                    'frames': frame_count,
                    'duration': duration,
                    'resolution': f"{width}x{height}"
                })
                cap.release()
        
        # Print statistics
        print(f"\n📊 Dataset Composition:")
        print(f"   Total: {stats['total_videos']} videos")
        print(f"   Real:  {stats['real_videos']} videos")
        print(f"   Fake:  {stats['fake_videos']} videos")
        
        print(f"\n🎬 Video Properties:")
        print(f"   Average FPS: {np.mean(stats['video_stats']['fps']):.2f}")
        print(f"   FPS Range: {np.min(stats['video_stats']['fps']):.1f} - {np.max(stats['video_stats']['fps']):.1f}")
        print(f"   Average Duration: {np.mean(stats['video_stats']['duration']):.2f} seconds")
        print(f"   Duration Range: {np.min(stats['video_stats']['duration']):.2f} - {np.max(stats['video_stats']['duration']):.2f} sec")
        
        return stats, video_infos
    
    def visualize_samples(self, df, num_samples=5):
        """Visualize sample frames from the dataset"""
        fig, axes = plt.subplots(2, num_samples, figsize=(20, 8))
        
        real_samples = df[df['type'] == 'real'].sample(min(num_samples, len(df[df['type'] == 'real'])))
        fake_samples = df[df['type'] == 'fake'].sample(min(num_samples, len(df[df['type'] == 'fake'])))
        
        # Show real videos
        for idx, (_, row) in enumerate(real_samples.iterrows()):
            cap = cv2.VideoCapture(row['path'])
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                axes[0, idx].imshow(frame_rgb)
                axes[0, idx].set_title(f"REAL: {Path(row['path']).name[:20]}")
                axes[0, idx].axis('off')
        
        # Show fake videos
        for idx, (_, row) in enumerate(fake_samples.iterrows()):
            cap = cv2.VideoCapture(row['path'])
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                axes[1, idx].imshow(frame_rgb)
                axes[1, idx].set_title(f"FAKE: {Path(row['path']).name[:20]}")
                axes[1, idx].axis('off')
        
        plt.tight_layout()
        plt.savefig('dataset_samples.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("✅ Saved visualization to 'dataset_samples.png'")
    
    def analyze_landmark_movement(self, df, num_videos_to_analyze=5):
        """Analyze landmark movement patterns in real vs fake videos"""
        print("\n" + "="*60)
        print("LANDMARK MOVEMENT ANALYSIS")
        print("="*60)
        
        if not DLIB_AVAILABLE:
            print("❌ Dlib not available. Install dlib for landmark analysis.")
            return None
        
        results = {
            'real': {'movement_stats': [], 'landmark_sequences': []},
            'fake': {'movement_stats': [], 'landmark_sequences': []}
        }
        
        # Analyze real videos
        real_videos = df[df['type'] == 'real'].head(num_videos_to_analyze)
        print("\n🔍 Analyzing REAL videos...")
        
        for _, row in real_videos.iterrows():
            landmarks_seq, faces = self.analyze_landmarks_over_sequence(row['path'], num_frames_to_analyze=20)
            valid_landmarks = [l for l in landmarks_seq if l is not None]
            
            if len(valid_landmarks) > 1:
                # Compute movement between consecutive frames
                movements = []
                for i in range(len(valid_landmarks) - 1):
                    diff = valid_landmarks[i+1] - valid_landmarks[i]
                    movement = np.mean(np.sqrt(np.sum(diff**2, axis=1)))
                    movements.append(movement)
                
                results['real']['movement_stats'].append({
                    'video': Path(row['path']).name,
                    'avg_movement': np.mean(movements) if movements else 0,
                    'max_movement': np.max(movements) if movements else 0,
                    'movement_std': np.std(movements) if movements else 0
                })
        
        # Analyze fake videos
        fake_videos = df[df['type'] == 'fake'].head(num_videos_to_analyze)
        print("\n🔍 Analyzing FAKE videos...")
        
        for _, row in fake_videos.iterrows():
            landmarks_seq, faces = self.analyze_landmarks_over_sequence(row['path'], num_frames_to_analyze=20)
            valid_landmarks = [l for l in landmarks_seq if l is not None]
            
            if len(valid_landmarks) > 1:
                movements = []
                for i in range(len(valid_landmarks) - 1):
                    diff = valid_landmarks[i+1] - valid_landmarks[i]
                    movement = np.mean(np.sqrt(np.sum(diff**2, axis=1)))
                    movements.append(movement)
                
                results['fake']['movement_stats'].append({
                    'video': Path(row['path']).name,
                    'avg_movement': np.mean(movements) if movements else 0,
                    'max_movement': np.max(movements) if movements else 0,
                    'movement_std': np.std(movements) if movements else 0
                })
        
        # Compare movements
        real_movements = [m['avg_movement'] for m in results['real']['movement_stats'] if m['avg_movement'] > 0]
        fake_movements = [m['avg_movement'] for m in results['fake']['movement_stats'] if m['avg_movement'] > 0]
        
        print(f"\n📊 Landmark Movement Comparison:")
        print(f"   Real videos - Avg movement: {np.mean(real_movements):.2f} pixels/frame")
        print(f"   Fake videos - Avg movement: {np.mean(fake_movements):.2f} pixels/frame")
        
        # Plot comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(['Real', 'Fake'], [np.mean(real_movements), np.mean(fake_movements)], 
               yerr=[np.std(real_movements), np.std(fake_movements)], capsize=10)
        ax.set_ylabel('Average Landmark Movement (pixels/frame)')
        ax.set_title('Landmark Movement Comparison: Real vs Fake Videos')
        ax.grid(True, alpha=0.3)
        plt.savefig('landmark_movement_comparison.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return results
    
    def save_dataset_info(self, df, output_file='ffpp_dataset_info.csv'):
        """Save dataset information to CSV"""
        df_info = df.copy()
        df_info['filename'] = df_info['path'].apply(lambda x: Path(x).name)
        df_info.to_csv(output_file, index=False)
        print(f"\n💾 Dataset info saved to '{output_file}'")
        return df_info
    
    def prepare_for_training(self, df, output_dir='processed_data'):
        """Prepare data for BMNet training"""
        print("\n" + "="*60)
        print("PREPARING DATA FOR BMNet TRAINING")
        print("="*60)
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save landmark extraction script
        with open(output_path / 'extract_landmarks.py', 'w') as f:
            f.write('''
import cv2
import dlib
import numpy as np
from pathlib import Path

def extract_landmarks_from_video(video_path, num_frames=30):
    """Extract 68 facial landmarks from video frames"""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    
    landmarks_sequence = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            
            if len(faces) > 0:
                landmarks = predictor(gray, faces[0])
                coords = []
                for i in range(68):
                    coords.append([landmarks.part(i).x, landmarks.part(i).y])
                landmarks_sequence.append(np.array(coords).flatten())
            else:
                landmarks_sequence.append(np.zeros(136))
        else:
            landmarks_sequence.append(np.zeros(136))
    
    cap.release()
    return np.array(landmarks_sequence)
''')
        
        print(f"✅ Data preparation scripts saved to '{output_dir}/'")
        print("\n🎯 Next Steps for BMNet Training:")
        print("   1. Download shape_predictor_68_face_landmarks.dat")
        print("   2. Run: python extract_landmarks.py")
        print("   3. Train BMNet model")

def main():
    """Main execution function"""
    print("="*60)
    print("FF++ DATASET EXPLORER")
    print("="*60)
    
    # Defaults to this project's local data folder:
    # data/original for real videos and data/Deepfakes for fake videos.
    default_dataset_path = Path(__file__).resolve().parents[1] / "data"
    user_path = input(f"\nEnter dataset path [{default_dataset_path}]: ").strip()
    DATASET_PATH = Path(user_path) if user_path else default_dataset_path
    
    if not DATASET_PATH.exists():
        print(f"❌ Path does not exist: {DATASET_PATH}")
        return
    
    # Initialize explorer
    explorer = FFPPDatasetExplorer(
        dataset_path=DATASET_PATH,
        num_videos=1000,
        fake_ratio=0.5
    )
    
    # Scan dataset
    num_real, num_fake = explorer.scan_dataset()
    
    if num_real == 0 or num_fake == 0:
        print("\n❌ Could not find sufficient videos. Please check:")
        print("   1. Dataset path is correct")
        print("   2. Dataset is properly downloaded")
        print("   3. You have read permissions")
        return
    
    # Select balanced subset
    df = explorer.select_balanced_subset()
    
    # Compute statistics
    stats, video_infos = explorer.compute_statistics(df)
    
    # Display sample video info
    print("\n📹 Sample Videos:")
    for info in video_infos[:5]:
        print(f"   {info['filename'][:30]:30} | {info['type']:5} | "
              f"{info['fps']:.1f}fps | {info['duration']:.1f}s")
    
    # Visualize samples
    explorer.visualize_samples(df, num_samples=4)
    
    # Analyze landmark movements (if dlib available)
    if DLIB_AVAILABLE:
        explorer.analyze_landmark_movement(df, num_videos_to_analyze=3)
    
    # Save dataset info
    explorer.save_dataset_info(df)
    
    # Prepare for training
    explorer.prepare_for_training(df)
    
    print("\n" + "="*60)
    print("✅ EXPLORATION COMPLETE!")
    print("="*60)
    print("\n📁 Generated Files:")
    print("   - ffpp_dataset_info.csv (dataset metadata)")
    print("   - dataset_samples.png (sample visualization)")
    print("   - processed_data/ (training preparation scripts)")
    if DLIB_AVAILABLE:
        print("   - landmark_movement_comparison.png (movement analysis)")
    
    return df, stats

if __name__ == "__main__":
    df, stats = main()
