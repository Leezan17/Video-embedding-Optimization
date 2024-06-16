import os
import shutil

# Define paths
data_path = 'UCF-101'  # Make sure this path is correct
train_list = os.path.join(data_path, 'ucfTrainTestlist', 'trainlist01.txt')
test_list = os.path.join(data_path, 'ucfTrainTestlist', 'testlist01.txt')
train_dir = os.path.join(data_path, 'train')
val_dir = os.path.join(data_path, 'val')

# Check if the train and test list files exist
if not os.path.exists(train_list):
    print(f"Train list file not found: {train_list}")
    exit(1)

if not os.path.exists(test_list):
    print(f"Test list file not found: {test_list}")
    exit(1)

# Create directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Function to copy files to the target directory
def copy_files(file_list, target_dir):
    with open(file_list, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.split()
            if len(parts) < 2:
                continue  # Skip lines that don't have the expected format
            video_path, _ = parts
            video_path = os.path.join(data_path, video_path)
            class_name = video_path.split('/')[1]
            class_dir = os.path.join(target_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            shutil.copy(video_path, class_dir)

# Copy training files
copy_files(train_list, train_dir)

# Copy validation files
copy_files(test_list, val_dir)

print("Dataset organized into train and val directories.")

