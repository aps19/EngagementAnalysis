import os
import subprocess
import pandas as pd
import cv2
from collections import defaultdict
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import math

# Expected OpenFace columns (you can adjust based on your needs)
OPENFACE_COLUMNS = ['frame', 'timestamp', 'confidence', 'success', 'pose_Tx', 'pose_Ty', 'pose_Tz', 'gaze_0_x', 'gaze_0_y', 'gaze_0_z']

def group_videos_by_subject(video_dir):
    """
    Group video files by subject and count how many videos each subject has.
    """
    video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    subject_videos = defaultdict(list)

    # Group video files by subject (assuming filenames are like subject_00*.mp4)
    for video_file in video_files:
        # Extract the subject identifier (e.g., 'subject_00')
        subject = video_file.split('_')[0] + '_' + video_file.split('_')[1]
        subject_videos[subject].append(video_file)

    return subject_videos

def get_video_frame_count(video_path):
    """
    Get the number of frames in a video using OpenCV.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return frame_count
    except Exception as e:
        print(f"Error reading video file {video_path}: {e}")
        return None

def is_csv_valid(csv_file, video_frame_count):
    """
    Check if the CSV file is valid:
    - The number of rows should match the number of frames in the video.
    - Required columns should be present.
    - No invalid or missing data in critical columns.
    """
    if not os.path.exists(csv_file):
        return False
    
    try:
        # Load the CSV file
        df = pd.read_csv(csv_file)

        # Check if the number of rows matches the video frame count
        if len(df) != video_frame_count:
            print(f"CSV row count {len(df)} does not match video frame count {video_frame_count}.")
            return False

        # Check if required columns are present
        if not all(column in df.columns for column in OPENFACE_COLUMNS):
            print(f"Missing required columns in {csv_file}.")
            return False

        # Check for invalid data (e.g., NaNs or missing data)
        if df[OPENFACE_COLUMNS].isnull().values.any():
            print(f"Invalid data found in {csv_file}.")
            return False

        return True
    except Exception as e:
        print(f"Error validating CSV file {csv_file}: {e}")
        return False

def is_video_processed(video_file, video_dir, output_dir):
    """
    Check if the video has already been processed by verifying the corresponding CSV file.
    Validate based on the number of frames and presence of required columns.
    """
    video_path = os.path.join(video_dir, video_file)
    output_csv = os.path.join(output_dir, os.path.splitext(video_file)[0] + ".csv")

    # Get the number of frames in the video
    video_frame_count = get_video_frame_count(video_path)
    if video_frame_count is None:
        return False

    # Check if the CSV file exists and is valid
    return is_csv_valid(output_csv, video_frame_count)

def run_openface_feature_extraction(subject, videos, video_dir, output_dir, openface_path, openface_home):
    """
    Run OpenFace's FeatureExtraction.exe on multiple videos for a subject in one batch.
    """
    video_paths = []
    output_csvs = []

    # Collect only the videos that haven't been processed or have invalid CSV files
    for video in videos:
        output_csv = os.path.join(output_dir, os.path.splitext(video)[0] + ".csv")
        if not is_video_processed(video, video_dir, output_dir):
            video_paths.append(os.path.join(video_dir, video))
            output_csvs.append(output_csv)

    # If all videos for the subject are already processed, skip execution
    if not video_paths:
        print(f"All videos for {subject} are already processed.")
        return

    # Construct the command for running multiple videos in one instance
    command = [openface_path]
    for video_path, output_csv in zip(video_paths, output_csvs):
        command.extend(["-f", video_path, "-of", output_csv])

    # Add other flags like pose, landmarks, etc.
    command.extend(["-out_dir", output_dir, "-pose", "-2Dfp", "-3Dfp", "-pdmparams", "-aus", "-gaze"])

    try:
        # Run the command from the OpenFace home directory
        subprocess.run(command, check=True, cwd=openface_home)
        print(f"Features extracted for {len(video_paths)} videos of {subject}")
    except subprocess.CalledProcessError as e:
        print(f"Error processing videos for {subject}: {e}")

def process_subjects_in_parallel(subject_videos, video_dir, output_dir, openface_path, openface_home, max_workers=6, batch_size=32):
    """
    Process videos for each subject in parallel using ProcessPoolExecutor.
    Each subject's videos are processed in a batch, with multiple processes running concurrently.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_subject = {}

        for subject, videos in subject_videos.items():
            # If there are too many videos, process them in sub-batches
            num_batches = math.ceil(len(videos) / batch_size)

            for i in range(num_batches):
                batch_videos = videos[i * batch_size: (i + 1) * batch_size]
                future = executor.submit(
                    run_openface_feature_extraction, subject, batch_videos, video_dir, output_dir, openface_path, openface_home
                )
                future_to_subject[future] = subject

        # Progress bar across subjects
        with tqdm(total=len(future_to_subject), desc="Processing subjects", unit="batch") as pbar:
            for future in as_completed(future_to_subject):
                subject = future_to_subject[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"Error with subject {subject}: {e}")
                pbar.update(1)  # Update progress bar after each batch is processed

def main():
    # Define the paths directly in the script
    video_directory = r"C:\Users\USER\abhishek\dataset\Train"  # Change this to the folder containing your video files
    output_directory = r"/mnt/d/Projects/engagement_prediction/processed_dataset"  # Change this to where you want the output files to be saved
    openface_feature_extraction_path = r"C:\Users\USER\abhishek\OpenFace_2.2.0_win_x64\FeatureExtraction.exe"  # Path to FeatureExtraction.exe
    openface_home_directory = r"/mnt/d/Projects/engagement_prediction/OpenFace_2.2.0_win_x64"  # Path to OpenFace home directory

    # Step 1: Group videos by subject and count them
    print("Counting videos per subject...")
    subject_videos = group_videos_by_subject(video_directory)

    # Step 2: Run Feature Extraction for each subject in parallel (batching for each subject)
    print("Running Feature Extraction on videos by subject in parallel with batching...")
    process_subjects_in_parallel(subject_videos, video_directory, output_directory, openface_feature_extraction_path, openface_home_directory, max_workers=8, batch_size=10)

if __name__ == "__main__":
    main()
