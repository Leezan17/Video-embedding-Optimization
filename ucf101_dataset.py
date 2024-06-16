# ucf101_dataset.py
import os
from torchvision.io import read_video
from torch.utils.data import Dataset

class UCF101Dataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.annotation_file = annotation_file
        self.transform = transform

        with open(annotation_file, 'r') as f:
            self.video_labels = f.readlines()

    def __len__(self):
        return len(self.video_labels)

    def __getitem__(self, idx):
        video_path = os.path.join(self.root_dir, self.video_labels[idx].split()[0])
        print(f"Loading video from path: {video_path}")  # Debugging line
        video = read_video(video_path, pts_unit='sec')[0]  # Only take the video frames
        if self.transform:
            video = self.transform(video)
        label = int(self.video_labels[idx].split()[1])
        return video, label
