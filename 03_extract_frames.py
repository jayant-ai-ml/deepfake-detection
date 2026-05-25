"""
Step 3: Extract frames from all videos (WITH RESUME CAPABILITY)
If extraction stops, you can restart and it will continue from where it left off
"""

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
from datetime import datetime

# ============================================
# CONFIGURATION
# ============================================

PREPARED_DATA_PATH = "data/prepared/"
PROCESSED_DATA_PATH = "data/processed/"
NUM_FRAMES = 30
FRAME_SIZE = (224, 224)  # Resize to this (H, W)

# ============================================
# PROGRESS TRACKING
# ============================================

class ProgressTracker:
    """Track which videos have been processed to enable resume"""
    
    def __init__(self, split_name, processed_data_path):
        self.split_name = split_name
        self.processed_data_path = processed_data_path
        self.progress_file = os.path.join(processed_data_path, f'{split_name}_progress.json')
        self.completed_indices = set()
        self.load_progress()
    
    def load_progress(self):
        """Load previously completed indices from file"""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    data = json.load(f)
                    self.completed_indices = set(data.get('completed_indices', []))
                print(f"   📌 Resume mode: Found {len(self.completed_indices)} already processed videos")
            except:
                print(f"   📌 Starting fresh: No previous progress found")
        else:
            print(f"   📌 Starting fresh: No previous progress found")
    
    def mark_completed(self, index):
        """Mark a video as completed"""
        self.completed_indices.add(int(index))
        self.save_progress()
    
    def save_progress(self):
        """Save progress to file"""
        with open(self.progress_file, 'w') as f:
            json.dump({
                'completed_indices': list(self.completed_indices),
                'last_updated': datetime.now().isoformat(),
                'total_completed': len(self.completed_indices)
            }, f)
    
    def is_completed(self, index):
        """Check if a video was already processed"""
        return int(index) in self.completed_indices
    
    def get_remaining_count(self, total):
        """Get number of videos remaining to process"""
        return total - len(self.completed_indices)
    
    def cleanup(self):
        """Delete progress file after successful completion"""
        if os.path.exists(self.progress_file):
            os.remove(self.progress_file)
            print(f"   🧹 Progress file cleaned up")

# ============================================
# FRAME EXTRACTION FUNCTION
# ============================================

def extract_frames_from_video(video_path, num_frames=30, target_size=(224, 224)):
    """
    Extract frames from a single video and resize them
    
    Returns:
        frames: numpy array of shape (num_frames, 224, 224, 3)
        success: True if successful, False otherwise
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None, False
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # If video is empty or too short
    if total_frames == 0:
        cap.release()
        return None, False
    
    # Calculate which frames to extract
    if total_frames < num_frames:
        # For short videos, repeat frames
        frame_indices = list(range(total_frames))
        # Pad with last frame
        while len(frame_indices) < num_frames:
            frame_indices.append(total_frames - 1)
    else:
        # Extract evenly spaced frames
        step = total_frames / num_frames
        frame_indices = [int(i * step) for i in range(num_frames)]
    
    frames = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize to target size
            frame_resized = cv2.resize(frame_rgb, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)
            frames.append(frame_resized)
        else:
            # If frame read fails, use black frame
            frames.append(np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8))
    
    cap.release()
    
    # Convert to numpy array
    frames_array = np.array(frames, dtype=np.uint8)
    
    return frames_array, True

# ============================================
# SAVE CHECKPOINT
# ============================================

def save_checkpoint(frames_list, labels_list, split_name, processed_data_path, is_final=False):
    """Save current progress as checkpoint"""
    
    checkpoint_frames_path = os.path.join(processed_data_path, f'{split_name}_checkpoint_frames.npy')
    checkpoint_labels_path = os.path.join(processed_data_path, f'{split_name}_checkpoint_labels.npy')
    
    # Convert to numpy arrays
    frames_array = np.array(frames_list, dtype=np.uint8)
    labels_array = np.array(labels_list)
    
    if is_final:
        # Final save - rename to final files
        final_frames_path = os.path.join(processed_data_path, f'{split_name}_frames.npy')
        final_labels_path = os.path.join(processed_data_path, f'{split_name}_labels.npy')
        
        # If checkpoint exists and we're doing final save, merge
        if os.path.exists(checkpoint_frames_path) and not is_final:
            existing_frames = np.load(checkpoint_frames_path)
            existing_labels = np.load(checkpoint_labels_path)
            frames_array = np.concatenate([existing_frames, frames_array])
            labels_array = np.concatenate([existing_labels, labels_array])
        
        np.save(final_frames_path, frames_array)
        np.save(final_labels_path, labels_array)
        
        # Delete checkpoint files
        if os.path.exists(checkpoint_frames_path):
            os.remove(checkpoint_frames_path)
        if os.path.exists(checkpoint_labels_path):
            os.remove(checkpoint_labels_path)
        
        print(f"\n   💾 Final save: {len(frames_array)} videos to {split_name}")
    else:
        # Checkpoint save - overwrite checkpoint file
        np.save(checkpoint_frames_path, frames_array)
        np.save(checkpoint_labels_path, labels_array)
        print(f"\n   💾 Checkpoint saved: {len(frames_array)} videos")

# ============================================
# LOAD CHECKPOINT
# ============================================

def load_checkpoint(split_name, processed_data_path):
    """Load previously saved checkpoint if exists"""
    
    checkpoint_frames_path = os.path.join(processed_data_path, f'{split_name}_checkpoint_frames.npy')
    checkpoint_labels_path = os.path.join(processed_data_path, f'{split_name}_checkpoint_labels.npy')
    
    if os.path.exists(checkpoint_frames_path) and os.path.exists(checkpoint_labels_path):
        print(f"   📦 Loading checkpoint...")
        frames = np.load(checkpoint_frames_path)
        labels = np.load(checkpoint_labels_path)
        print(f"   Loaded {len(frames)} videos from checkpoint")
        return list(frames), list(labels)
    
    return [], []

# ============================================
# PROCESS ONE SPLIT WITH RESUME
# ============================================

def process_split_with_resume(split_name):
    """
    Process all videos in a split with resume capability
    """
    
    csv_path = os.path.join(PREPARED_DATA_PATH, f'{split_name}.csv')
    
    if not os.path.exists(csv_path):
        print(f"⚠️ {csv_path} not found. Skipping {split_name}.")
        return False
    
    df = pd.read_csv(csv_path)
    
    print(f"\n{'='*60}")
    print(f"Processing {split_name.upper()} split")
    print(f"{'='*60}")
    print(f"Total videos: {len(df)}")
    print(f"Target frame size: {FRAME_SIZE[0]}x{FRAME_SIZE[1]}")
    print(f"Frames per video: {NUM_FRAMES}")
    
    # Initialize progress tracker
    tracker = ProgressTracker(split_name, PROCESSED_DATA_PATH)
    
    # Load any existing checkpoint
    all_frames, all_labels = load_checkpoint(split_name, PROCESSED_DATA_PATH)
    existing_count = len(all_frames)
    
    if existing_count > 0:
        print(f"   ✅ Resuming from checkpoint: {existing_count} videos already extracted")
    
    # Track failed videos
    failed_videos = []
    
    # Process each video with progress bar
    remaining = tracker.get_remaining_count(len(df))
    
    if remaining == 0:
        print(f"   ✅ All {len(df)} videos already processed!")
        # Final save from checkpoint
        if existing_count > 0:
            save_checkpoint(all_frames, all_labels, split_name, PROCESSED_DATA_PATH, is_final=True)
        tracker.cleanup()
        return True
    
    # Create progress bar for remaining videos
    pbar = tqdm(total=remaining, desc=f"Extracting {split_name}", unit="video")
    
    for idx, row in df.iterrows():
        # Skip if already processed
        if tracker.is_completed(idx):
            pbar.update(1)
            continue
        
        video_path = row['video_path']
        label = row['label']
        
        try:
            frames, success = extract_frames_from_video(video_path, NUM_FRAMES, FRAME_SIZE)
            
            if success and frames is not None:
                all_frames.append(frames)
                all_labels.append(label)
                tracker.mark_completed(idx)
                
                # Save checkpoint every 50 videos
                if len(all_frames) % 50 == 0:
                    save_checkpoint(all_frames, all_labels, split_name, PROCESSED_DATA_PATH, is_final=False)
            else:
                failed_videos.append({
                    'index': idx,
                    'path': video_path,
                    'error': 'Extraction failed'
                })
                tracker.mark_completed(idx)  # Mark as completed to avoid retrying
                
        except Exception as e:
            print(f"\n   ⚠️ Error on video {idx}: {e}")
            failed_videos.append({
                'index': idx,
                'path': video_path,
                'error': str(e)
            })
            tracker.mark_completed(idx)
        
        pbar.update(1)
    
    pbar.close()
    
    # Final save
    if len(all_frames) > 0:
        save_checkpoint(all_frames, all_labels, split_name, PROCESSED_DATA_PATH, is_final=True)
    
    # Save failed videos list
    if failed_videos:
        failed_path = os.path.join(PROCESSED_DATA_PATH, f'{split_name}_failed.json')
        with open(failed_path, 'w') as f:
            json.dump(failed_videos, f, indent=2)
        print(f"   ⚠️ Failed videos: {len(failed_videos)}")
        print(f"   Failed list saved to: {failed_path}")
    
    # Print summary
    final_frames_path = os.path.join(PROCESSED_DATA_PATH, f'{split_name}_frames.npy')
    if os.path.exists(final_frames_path):
        final_frames = np.load(final_frames_path)
        final_labels = np.load(os.path.join(PROCESSED_DATA_PATH, f'{split_name}_labels.npy'))
        
        print(f"\n✅ {split_name.upper()} saved successfully!")
        print(f"   Frames shape: {final_frames.shape}")
        print(f"   Labels shape: {final_labels.shape}")
        print(f"   Memory used: {final_frames.nbytes / (1024**3):.2f} GB")
        print(f"   REAL videos: {np.sum(final_labels == 0)}")
        print(f"   FAKE videos: {np.sum(final_labels == 1)}")
    
    # Cleanup progress file on success
    if len(failed_videos) == 0:
        tracker.cleanup()
    
    return True

# ============================================
# VERIFICATION
# ============================================

def verify_extraction():
    """Verify that files were saved correctly"""
    
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)
    
    all_good = True
    
    for split in ['train', 'val', 'test']:
        frames_path = os.path.join(PROCESSED_DATA_PATH, f'{split}_frames.npy')
        labels_path = os.path.join(PROCESSED_DATA_PATH, f'{split}_labels.npy')
        
        if os.path.exists(frames_path) and os.path.exists(labels_path):
            frames = np.load(frames_path)
            labels = np.load(labels_path)
            
            print(f"\n✅ {split.upper()} set:")
            print(f"   Videos: {len(frames)}")
            print(f"   Frames per video: {frames.shape[1]}")
            print(f"   Frame shape: {frames.shape[2:]}")
            print(f"   Data type: {frames.dtype}")
            print(f"   File size: {os.path.getsize(frames_path) / (1024**2):.2f} MB")
            print(f"   REAL: {np.sum(labels == 0)}")
            print(f"   FAKE: {np.sum(labels == 1)}")
        else:
            print(f"\n❌ {split.upper()} set: NOT FOUND")
            all_good = False
    
    return all_good

# ============================================
# SAMPLE VISUALIZATION
# ============================================

def visualize_sample():
    """Display a sample video's frames"""
    try:
        import matplotlib.pyplot as plt
        
        frames_path = os.path.join(PROCESSED_DATA_PATH, 'train_frames.npy')
        
        if not os.path.exists(frames_path):
            print("\n⚠️ No train_frames.npy found to visualize")
            return
        
        frames = np.load(frames_path)
        
        if len(frames) == 0:
            print("\n⚠️ No frames to visualize")
            return
        
        # Take first video
        video_frames = frames[0]
        
        # Display first 6 frames
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.ravel()
        
        for i in range(min(6, len(video_frames))):
            axes[i].imshow(video_frames[i])
            axes[i].set_title(f"Frame {i+1}")
            axes[i].axis('off')
        
        os.makedirs("results", exist_ok=True)
        plt.suptitle(f"Sample Frames (Resized to {FRAME_SIZE[0]}x{FRAME_SIZE[1]})", fontsize=14)
        plt.savefig("results/sample_frames.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        print("\n✅ Sample visualization saved to results/sample_frames.png")
        
    except ImportError:
        print("\n⚠️ Matplotlib not installed. Run: pip install matplotlib")
    except Exception as e:
        print(f"\n⚠️ Could not visualize: {e}")

# ============================================
# CLEANUP FUNCTION (Optional)
# ============================================

def cleanup_checkpoints():
    """Delete all checkpoint files to start fresh"""
    
    print("\n🧹 Cleaning up checkpoint files...")
    
    checkpoint_files = [f for f in os.listdir(PROCESSED_DATA_PATH) if 'checkpoint' in f]
    
    for file in checkpoint_files:
        file_path = os.path.join(PROCESSED_DATA_PATH, file)
        os.remove(file_path)
        print(f"   Removed: {file}")
    
    progress_files = [f for f in os.listdir(PROCESSED_DATA_PATH) if f.endswith('_progress.json')]
    
    for file in progress_files:
        file_path = os.path.join(PROCESSED_DATA_PATH, file)
        os.remove(file_path)
        print(f"   Removed: {file}")
    
    print("   ✅ Cleanup complete!")

# ============================================
# MAIN
# ============================================

def main():
    print("="*60)
    print("FRAME EXTRACTION WITH RESUME CAPABILITY")
    print("="*60)
    
    # Create output directory
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    
    # Ask if user wants to cleanup old checkpoints
    checkpoint_exists = any(['checkpoint' in f for f in os.listdir(PROCESSED_DATA_PATH)]) if os.path.exists(PROCESSED_DATA_PATH) else False
    
    if checkpoint_exists:
        print("\n⚠️ Found existing checkpoint files!")
        choice = input("   Do you want to (r)esume, (c)leanup and start fresh, or (s)kip this split? (r/c/s): ").lower()
        
        if choice == 'c':
            cleanup_checkpoints()
        elif choice == 's':
            print("   Skipping...")
            return
        else:
            print("   Resuming from checkpoints...")
    
    # Process each split
    splits = ['train', 'val', 'test']
    
    for split in splits:
        success = process_split_with_resume(split)
        if not success:
            print(f"\n⚠️ Failed to process {split}. Continuing with next split...")
    
    # Verify all files were saved
    all_verified = verify_extraction()
    
    # Visualize sample if verification passed
    if all_verified:
        visualize_sample()
    else:
        print("\n⚠️ Verification failed. Please check the errors above.")
    
    print("\n" + "="*60)
    if all_verified:
        print("✅ FRAME EXTRACTION COMPLETE!")
        print("\n📋 Next step: Run 04_extract_faces.py")
    else:
        print("❌ FRAME EXTRACTION INCOMPLETE!")
        print("   Run the script again - it will resume from where it stopped!")
    print("="*60)

if __name__ == "__main__":
    main()