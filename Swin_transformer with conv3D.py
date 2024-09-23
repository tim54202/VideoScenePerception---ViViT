import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from ptflops import get_model_complexity_info
import timm
from dataset_1stream_singleppl import GCRFDataset
import glob
torch.cuda.set_device(0)

import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
from timm.models.swin_transformer import WindowAttention
import matplotlib.pyplot as plt
import torchvision.models.video as video_models
import time

class ResNet3D18(nn.Module):
    def __init__(self, num_classes, num_frames=32):
        super(ResNet3D18, self).__init__()
        
        # Load pre-trained 3D ResNet18 model
        self.base_model = video_models.r3d_18(pretrained=True)
        
        # Modify the first convolution layer to accept 3 input channels
        self.base_model.stem[0] = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), 
                                            stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        
        # Replace the last fully connected layer
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(in_features, num_classes)
        
        # Add dropout
        self.dropout = nn.Dropout(0.7)
        
    def forward(self, x):
        print("Input shape:", x.shape)
        # x shape: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        
        # Rearrange dimensions to fit ResNet3D
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (B, C, T, H, W)
        
        # Pass through base ResNet3D model
        features = self.base_model.stem(x)
        print("Output shape of stem:", features.shape)
        
        features = self.base_model.layer1(features)
        print("Output shape of layer1:", features.shape)
        
        features = self.base_model.layer2(features) 
        print("Output shape of layer2:", features.shape)
        
        features = self.base_model.layer3(features)
        print("Output shape of layer3:", features.shape)
        
        features = self.base_model.layer4(features)
        print("Output shape of layer4:", features.shape)
        
        # Global average pooling
        features = self.base_model.avgpool(features)
        features = features.view(B, -1)  # (B, D)
        
        # Dropout
        features = self.dropout(features)
        
        # Classification
        output = self.base_model.fc(features)
        
        return output

        #13.87 GMac, 33.14 M
 

# Modified VideoSwinTransformerWithTokens class
class VideoSwinTransformerWithTokens(nn.Module):
    def __init__(self, base_model, num_classes, num_frames=32, input_channels=96):
        super(VideoSwinTransformerWithTokens, self).__init__()
        self.conv3d_up = nn.Conv3d(3, 96, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv3d_down = nn.Conv3d(96, 3, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        
        self.base_model = base_model
        self.num_frames = num_frames
        
        # Get the output dimension of Swin Transformer
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            dummy_output = self.base_model(dummy_input)
            swin_output_dim = dummy_output.shape[-1]
        
        # Temporal encoder
        self.temporal_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=swin_output_dim, nhead=8),
            num_layers=1
        )
        
        # Classifier
        self.classifier = nn.Linear(swin_output_dim, num_classes)
        self.dropout = nn.Dropout(0.7)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.transpose(1, 2)  # (B, C, T, H, W)
        
        # 3D convolution processing
        x = self.conv3d_up(x)
        x = self.conv3d_down(x)  # Output shape (B, 3, T, H, W)
        
        # Reshape to fit Swin Transformer
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # Convert to (B, T, C, H, W)
        spatial_features = []
        for t in range(T):
            frame = x[:, t, :, :, :]  # (B, C, H, W)
            frame_features = self.base_model(frame)  # (B, D)
            spatial_features.append(frame_features)
        spatial_features = torch.stack(spatial_features, dim=1)  # (B, T, D)
        
        # Temporal encoding
        temporal_features = self.temporal_encoder(spatial_features.transpose(0, 1)).transpose(0, 1)  # Output (B, T, D)
        
        # Combine temporal and spatial features
        combined_features = temporal_features.mean(1)  # Average operation (B, D)
        
        # Apply dropout before classification
        combined_features = self.dropout(combined_features)
        
        # Classification
        output = self.classifier(combined_features)
        return output

        #499.47 GMac, 95.15M

class VideoSwinTransformerSingleEncoder(nn.Module):
    def __init__(self, base_model, num_classes, num_frames=32, input_channels=96):
        super(VideoSwinTransformerSingleEncoder, self).__init__()
        self.conv3d_up = nn.Conv3d(3, 96, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv3d_down = nn.Conv3d(96, 3, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        self.base_model = base_model
        self.num_frames = num_frames
        
        # Get the output dimension of Swin Transformer
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            dummy_output = self.base_model.forward_features(dummy_input)
            swin_output_dim = dummy_output.shape[-1]
        
        # Classifier
        self.classifier = nn.Linear(swin_output_dim, num_classes)
        self.dropout = nn.Dropout(0.7)

    def forward(self, x):
        print("Input shape:", x.shape)
        B, T, C, H, W = x.shape
        x = x.transpose(1, 2)  # (B, C, T, H, W)
        
        # 3D convolution processing: 3 -> 96 -> 3
        x = self.conv3d_up(x)  # Output shape (B, 96, T, H, W)
        x = self.conv3d_down(x)  # Output shape (B, 3, T, H, W)
        
        # Reshape to fit Swin Transformer
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # Convert to (B, T, C, H, W)
        spatial_features = []
        for t in range(T):
            frame = x[:, t, :, :, :]  # (B, C, H, W)
            frame_features = self.base_model(frame)  # (B, D)
            spatial_features.append(frame_features)
        spatial_features = torch.stack(spatial_features, dim=1)  # (B, T, D)
        
        # Temporal dimension average pooling
        combined_features = spatial_features.mean(1)  # Average operation (B, D)
        
        # Apply dropout before classification
        combined_features = self.dropout(combined_features)
        
        # Classification
        output = self.classifier(combined_features)
        return output

        #499.2 GMac, 86.75M


# Modified ExGAN class
class ExGAN:
    def __init__(self, model_name):
        self.batch_size = 8 
        self.n_epochs = 8
        self.img_height = 224
        self.img_width = 224
        self.channels = 3
        self.num_frames = 32
        self.c_dim = 2  # Assuming binary classification
        self.model_name = model_name

        self.lr = 0.00005
        self.b1 = 0.5
        self.b2 = 0.999
        self.log_write = open(f"./log_{self.model_name}_results.txt", "w")

        self.criterion_cls = nn.CrossEntropyLoss().cuda()

        self.training_loss = []
        self.testing_loss = []

        if self.model_name == "Swin-Transformer_single":
            base_model = timm.create_model("swin_base_patch4_window7_224_in22k", pretrained=True, num_classes=0)
            self.model = VideoSwinTransformerSingleEncoder(base_model, num_classes=self.c_dim, num_frames=self.num_frames).cuda()

        elif self.model_name == "Swin-Transformer-base1":
            base_model = timm.create_model("swin_base_patch4_window7_224_in22k", pretrained=True, num_classes=0)
            self.model = VideoSwinTransformerWithTokens(base_model, num_classes=self.c_dim, num_frames=self.num_frames).cuda()

        elif self.model_name == "ViT":
            base_model = timm.create_model("vit_tiny_patch16_224", pretrained=True, num_classes=0)
            self.model = ViTWith3DConv(base_model, num_classes=self.c_dim, num_frames=self.num_frames).cuda()

        elif self.model_name == "ResNet3D18":
            self.model = ResNet3D18(num_classes=self.c_dim, num_frames=self.num_frames).cuda()


        total = sum([param.nelement() for param in self.model.parameters()])
        print(f"{self.model_name}-Number of parameter: {total / 1e6:.4f}M")

        try:
            macs, params = get_model_complexity_info(
                self.model, 
                (self.num_frames, 3,  224, 224), 
                as_strings=True,
                print_per_layer_stat=True, 
                verbose=True
            )
            print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
            print('{:<30}  {:<8}'.format('Number of parameters: ', params))
        except Exception as e:
            print(f"Unable to estimate FLOPs due to: {str(e)}")
            print("Continuing without FLOPs estimation.")

        # Use Adam optimizer with weight decay
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2), weight_decay=1e-5)
        
        # Use ReduceLROnPlateau to dynamically adjust learning rate
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=2, verbose=True)

        self.Transform = transforms.Compose([
            transforms.Resize((self.img_height, self.img_width)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.dataloader = DataLoader(
            GCRFDataset(root='/home/ubuntu/tim/train/dataset/train',
                        transforms_=self.Transform, mode='train', num_frames=self.num_frames),
            batch_size=self.batch_size,  
            shuffle=True,
            num_workers=2
        )

        self.test_dataloader = DataLoader(
            GCRFDataset(root='/home/ubuntu/tim/train/dataset/test',
                        transforms_=self.Transform, mode='train', num_frames=self.num_frames),
            batch_size=self.batch_size,  
            shuffle=False,
            num_workers=2
        )



    def train(self):
        print(f'Train on the {self.model_name} model')
        start = time.time()
        best_accuracy = 0
        patience = 3
        no_improve = 0
        
        for epoch in range(self.n_epochs):
            self.model.train()
            running_loss = 0.0
            running_corrects = 0
            total_samples = 0

            for batch, (videos, labels) in enumerate(self.dataloader):
                videos, labels = videos.cuda(), labels.cuda()
                
                self.optimizer.zero_grad()
                
                outputs = self.model(videos)
                
                loss = self.criterion_cls(outputs, labels)
                loss.backward()
                self.optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * videos.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total_samples += videos.size(0)

                # Print training progress every 10 batches
                if batch % 10 == 0:
                    print(f"Epoch {epoch}/{self.n_epochs - 1}, Batch {batch}, "
                          f"Train Loss: {running_loss / total_samples:.4f}, "
                          f"Train ACC: {100.0 * running_corrects / total_samples:.4f}%")

            # Calculate epoch statistics
            epoch_loss = running_loss / total_samples
            epoch_acc = 100.0 * running_corrects / total_samples

            self.training_loss.append(epoch_loss)

            print(f"Epoch {epoch} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}%")
            
            # Evaluate on test set
            test_acc = self.test()
            self.log_write.write(f"{epoch}\t{epoch_loss:.4f}\t{epoch_acc:.4f}\t{test_acc:.4f}\n")
            self.scheduler.step(test_acc)

            # Save best model and check for early stopping
            if test_acc > best_accuracy:
                best_accuracy = test_acc
                torch.save(self.model.state_dict(), f"./model_{self.model_name}_0806.pth")
                no_improve = 0
            else:
                no_improve += 1
            
            # Early stopping
            if no_improve >= patience:
                print(f"Early stopping triggered. No improvement for {patience} epochs.")
                break

        end = time.time()  
        elapsed_time = end - start 
        print(f'Training completed in: {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s')
        print(f'The best accuracy is: {best_accuracy:.4f}')
        self.log_write.write(f'The best accuracy is: {best_accuracy:.4f}\n')
        self.log_write.close()

    def test(self):
        print(f'Testing on the {self.model_name} model')
        self.model.eval()
        running_corrects = 0
        total_samples = 0
        running_loss = 0.0

        with torch.no_grad():
            for videos, labels in self.test_dataloader:
                videos, labels = videos.cuda(), labels.cuda()
                
                outputs = self.model(videos)

                loss = self.criterion_cls(outputs, labels)
                
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)
                total_samples += videos.size(0)
                running_loss += loss.item() * videos.size(0)

        # Calculate test statistics
        test_loss = running_loss / total_samples
        test_acc = 100.0 * running_corrects / total_samples

        self.testing_loss.append(test_loss)

        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}%')
        return test_acc

    def plot_loss(self):
        epochs = range(1, len(self.training_loss) + 1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.training_loss, label='Training Loss')
        plt.plot(epochs, self.testing_loss, label='Testing Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'{self.model_name} Training and Testing Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

def main():
    models_list = ['ResNet3D18']
    for model in models_list:
        exgan = ExGAN(model)
        exgan.train()
        exgan.plot_loss()

if __name__ == "__main__":
    main()
