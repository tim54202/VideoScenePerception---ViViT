import os
import glob
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class GCRFDataset(Dataset):
    def __init__(self, root, transforms_=None, mode='train', split_ratio=1, num_frames=32):
        self.transform = transforms_
        self.root = root
        self.folders = []
        self.labels = []
        self.num_frames = num_frames

        # Define paths for good and bad samples
        good_path = os.path.join(root, 'good')
        bad_path = os.path.join(root, 'bad')
        
        # Get sorted lists of good and bad folders
        good_folders = sorted([os.path.join(good_path, d) for d in os.listdir(good_path) if os.path.isdir(os.path.join(good_path, d))])
        bad_folders = sorted([os.path.join(bad_path, d) for d in os.listdir(bad_path) if os.path.isdir(os.path.join(bad_path, d))])

        # Combine good and bad folders with their labels
        all_folders = [(folder, 1) for folder in good_folders] + [(folder, 0) for folder in bad_folders]

        # Shuffle and split the data based on the mode (train or test)
        random.shuffle(all_folders)
        split_index = int(len(all_folders) * split_ratio)
        selected_folders = all_folders[:split_index] if mode == 'train' else all_folders[split_index:]

        # Populate the folders and labels lists
        for folder, label in selected_folders:
            for part in ['part_1', 'part_2', 'part_3']:
                for subfolder in ['folder_1', 'folder_2', 'folder_3', 'folder_4']:
                    part_folder = os.path.join(folder, part, subfolder)
                    if os.path.exists(part_folder):
                        self.folders.append(part_folder)
                        self.labels.append(label)

    def __getitem__(self, index):
        folder = self.folders[index]
        label = self.labels[index]
        images = []

        # Get sorted list of image files in the folder
        image_files = sorted(glob.glob(os.path.join(folder, '*.jpg')))
        
        # Adjust the number of images to match num_frames
        if len(image_files) < self.num_frames:
            image_files = (image_files * ((self.num_frames // len(image_files)) + 1))[:self.num_frames]
        elif len(image_files) > self.num_frames:
            image_files = random.sample(image_files, self.num_frames)

        # Load and transform images
        for image_file in image_files:
            img = Image.open(image_file).convert('RGB')
            if self.transform:
                img = self.transform(img)
            images.append(img)

        # Stack images into a tensor
        images_tensor = torch.stack(images)

        return images_tensor, label

    def __len__(self):
        return len(self.folders)

# Test dataset loading
train_dataset = GCRFDataset(root='/home/ubuntu/tim/train/dataset/train', transforms_=transforms.ToTensor(), mode='train', num_frames=32)
test_dataset = GCRFDataset(root='/home/ubuntu/tim/train/dataset/test', transforms_=transforms.ToTensor(), mode='train', num_frames=32)

print(f'Total training samples: {len(train_dataset)}')
print(f'Total testing samples: {len(test_dataset)}')
