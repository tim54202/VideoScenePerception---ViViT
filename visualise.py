import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import os
import glob
import timm
from implement import VideoSwinTransformerWithTokens, ExGAN
from timm.models.swin_transformer import WindowAttention
import torch.nn.functional as F
import math
from scipy.special import softmax

class VisualizationTool:
    def __init__(self, model_path, num_classes=2, num_frames=32, img_size=224):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(model_path, num_classes, num_frames)
        self.num_frames = num_frames
        self.img_size = 224

    def load_model(self, model_path, num_classes, num_frames):
        # Load the base model and create a VideoSwinTransformerWithTokens instance
        base_model = timm.create_model("swin_base_patch4_window7_224_in22k", pretrained=True, num_classes=0)
        model = VideoSwinTransformerWithTokens(base_model, num_classes=num_classes, num_frames=num_frames).to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device), strict=False)
        model.eval()
        return model

    def preprocess_video(self, video_path):
        # Define image transformations
        transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        frames = []
        image_files = sorted(glob.glob(os.path.join(video_path, '*.jpg')))[:self.num_frames]
        
        # Process each frame
        for img_path in image_files:
            img = Image.open(img_path).convert('RGB')
            img = transform(img)
            frames.append(img)
        
        video_tensor = torch.stack(frames).unsqueeze(0).to(self.device)  # (1, T, C, H, W)
        return video_tensor

    def print_model_structure(self):
        # Print the structure of the model
        print("Model structure:")
        for name, module in self.model.named_modules():
            print(f"{name}: {module.__class__.__name__}")

    def grad_cam_3d(self, input_tensor, target_class):
        self.model.zero_grad()
        
        # Use conv3d_down as the target layer
        target_layer = self.model.conv3d_down
        
        # Register hooks
        activations = []
        gradients = []
        def forward_hook(module, input, output):
            activations.append(output)
        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])
        
        handle_forward = target_layer.register_forward_hook(forward_hook)
        handle_backward = target_layer.register_backward_hook(backward_hook)
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Backward pass for the target class
        self.model.zero_grad()
        output[0, target_class].backward()
        
        # Remove hooks
        handle_forward.remove()
        handle_backward.remove()
        
        # Calculate Grad-CAM++
        activation = activations[0].squeeze()  # [3, T, H, W]
        gradient = gradients[0].squeeze()  # [3, T, H, W]
        
        alpha_num = gradient.pow(2)
        alpha_denom = alpha_num.mul(2) + activation.mul(gradient.pow(3)).sum(dim=[1,2,3], keepdim=True)
        alpha = alpha_num.div(alpha_denom + 1e-7)
        
        weights = (alpha * F.relu(gradient)).sum(dim=[1,2,3], keepdim=True)
        cam = (weights * activation).sum(dim=0)  # [T, H, W]
        
        cam = F.relu(cam)
        
        # Normalize based on global max and min values
        cam = torch.nn.functional.normalize(cam, dim=(1, 2), p=2, eps=1e-12)
        
        # Apply non-linear transformation to highlight the most important regions
        cam = torch.sigmoid(cam * 10)
        
        # Calculate average activation for each frame
        frame_importance = torch.mean(cam, dim=(1, 2))  # [T]
        frame_importance = F.softmax(frame_importance, dim=0)
        
        return cam

    def plot_frame_importance(self, cam):
        # Extract importance values
        importance = cam.squeeze().cpu().detach().numpy()
        frame_importance = np.mean(importance[1:-1], axis=(1, 2))
        frame_importance = softmax(frame_importance)
        print("Range of frame_importance :", np.min(frame_importance), np.max(frame_importance))
        
        # Create frame indices
        frames = range(1, len(importance) - 1)
        
        # Create the plot
        plt.figure(figsize=(12, 6))
        plt.plot(frames, frame_importance, marker='o')
        plt.title('Dual Encoder: Frame Importance Over Time')
        plt.xlabel('Frame Index')
        plt.ylabel('Importance Score')
        plt.grid(True)
        
        # Annotate the most important frame
        max_frame = frame_importance.argmax()
        plt.annotate(f'Main Frame', 
                     xy=(max_frame, frame_importance[max_frame]), 
                     xytext=(5, 5), 
                     textcoords='offset points',
                     ha='left', 
                     va='bottom',
                     bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.savefig('/home/ubuntu/tim/train/result/dual/importance_over_time.png')

    def plot_3d_frame_importance(self, cam):
        # Create a 3D plot of frame importance
        if isinstance(cam, tuple):
            cam = cam[0]
        importance = cam.squeeze().cpu().detach().numpy()
        frame_importance = np.mean(importance, axis=(1, 2))
        
        x = np.arange(len(frame_importance))
        y = np.arange(1)
        X, Y = np.meshgrid(x, y)
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        surf = ax.plot_surface(X, Y, frame_importance.reshape(1, -1), cmap='viridis')
        
        ax.set_xlabel('Frame Index')
        ax.set_ylabel('Y')
        ax.set_zlabel('Importance Score')
        ax.set_title('Dual Encoder: Frame Importance Over Time')
        
        fig.colorbar(surf, shrink=0.5, aspect=5)
        
        plt.show()

    def get_attention_maps(self, x, main_frame_index):
        # Get attention maps for the main frame
        self.model.eval()
        self.attn_maps = []
        
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            if output.dim() == 4:  # (B, C, H, W)
                output = output.permute(0, 2, 3, 1)  # Convert to (B, H, W, C)
            output = output.reshape(output.shape[0], -1, output.shape[-1])  # Reshape to (B, N, C)
            print(f"Hook output shape: {output.shape}")
            self.attn_maps.append((output.detach().cpu().numpy(), output.shape[1], output.shape[2]))

        
        hooks = []
        for name, module in self.model.named_modules():
            if "blocks" in name and name.endswith("norm2"):
                hook = module.register_forward_hook(hook_fn)
                hooks.append(hook)
                print(f"Registered hook for: {name}")
        
        # Use the main frame as input
        main_frame = x[:, main_frame_index].unsqueeze(1)
        self.model(main_frame)
        
        for hook in hooks:
            hook.remove()
        
        return self.attn_maps

    def get_all_attention_maps(self, x):
        # Get attention maps for all frames
        self.model.eval()
        all_frame_attn_maps = []
        
        for frame in range(x.shape[1]):
            print(f"Processing frame {frame}")
            frame_attn_maps = []
            
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    attn_weights = output
                else:
                    attn_weights = output
                
                if attn_weights is not None:
                    print(f"Layer: {module.__class__.__name__}, Attention Map shape: {attn_weights.shape}")
                    frame_attn_maps.append(attn_weights.detach().cpu().numpy())
            
            hooks = []
            for name, module in self.model.named_modules():
                if isinstance(module, WindowAttention):
                    hooks.append(module.register_forward_hook(hook_fn))
            
            with torch.no_grad():
                _ = self.model(x[:, frame:frame+1])
            
            for hook in hooks:
                hook.remove()
            
            all_frame_attn_maps.append(frame_attn_maps)
        
        return all_frame_attn_maps

    def compute_attention_rollout(self, attn_maps):
        # Compute attention rollout for all frames
        frame_rollouts = []
        
        for frame_attn_maps in attn_maps:
            accumulated_rollout = None
            
            for layer_idx, layer_attn_map in enumerate(frame_attn_maps):
                print(f"Processing Layer {layer_idx}, Shape: {layer_attn_map.shape}")
                
                # Process attention maps based on their shape
                if layer_attn_map.shape == (64, 49, 128):  # 56x56
                    attn = layer_attn_map.mean(axis=0)  # Average over all heads
                    attn = attn @ attn.T  # Compute self-attention
                elif layer_attn_map.shape == (16, 49, 256):  # 28x28
                    attn = layer_attn_map.mean(axis=0)
                    attn = attn @ attn.T
                elif layer_attn_map.shape == (4, 49, 512):  # 14x14
                    attn = layer_attn_map.mean(axis=0)
                    attn = attn @ attn.T
                elif layer_attn_map.shape == (1, 49, 1024):  # 7x7
                    attn = layer_attn_map.squeeze(0)
                    attn = attn @ attn.T
                else:
                    print(f"Warning: Skipping Layer {layer_attn_map.shape}")
                    continue
                
                # Add residual connections and normalize
                attn = 0.5 * attn + 0.5 * np.eye(attn.shape[0])
                attn /= attn.sum(axis=-1, keepdims=True)
                
                if accumulated_rollout is None:
                    accumulated_rollout = attn
                else:
                    # Ensure dimensions match
                    if accumulated_rollout.shape != attn.shape:
                        accumulated_rollout = np.array(Image.fromarray(accumulated_rollout).resize(
                            (attn.shape[1], attn.shape[0]), Image.BICUBIC))
                    accumulated_rollout = attn @ accumulated_rollout
            
            if accumulated_rollout is not None:
                rollout = accumulated_rollout
                rollout = (rollout - rollout.min()) / (rollout.max() - rollout.min())
                
                # Upsample to original image size (224x224)
                rollout = np.array(Image.fromarray((rollout * 255).astype(np.uint8)).resize((224, 224), Image.BICUBIC))
                rollout = rollout.astype(np.float32) / 255.0
                
                frame_rollouts.append(rollout)
            else:
                print(f"Warning: No rollout computed for this frame")
        
        return frame_rollouts
    

    def compute_and_visualize_attention_rollout(self, all_frame_attn_maps, original_frames, img_size=224):
        # Compute and visualize attention rollout for all frames
        num_frames = len(all_frame_attn_maps)
        frame_rollouts = []

        for frame, frame_attn_maps in enumerate(all_frame_attn_maps):
            combined_attn_map = self.combine_attention_maps(frame_attn_maps, (img_size, img_size))
            frame_rollouts.append(combined_attn_map)

        self.visualize_attention_rollouts(frame_rollouts, original_frames, "Dual Encoder Attention Rollout Map")


    def combine_attention_maps(self, attn_maps, img_shape):
        combined_attn_map = None
        for attn_map in attn_maps:
            if attn_map.ndim == 4:  # [batch, heads, height, width]
                avg_attn_map = np.mean(attn_map, axis=(0, 1))  # Average over batch and heads
            elif attn_map.ndim == 3:  # [heads, height, width]
                avg_attn_map = np.mean(attn_map, axis=0)  # Average over heads
            else:
                avg_attn_map = attn_map

            print(f"Attention Shape: {avg_attn_map.shape}")  

            if combined_attn_map is None:
                combined_attn_map = avg_attn_map
            else:
                # Make sure the size of two images
                if combined_attn_map.shape != avg_attn_map.shape:
                    avg_attn_map = np.resize(avg_attn_map, combined_attn_map.shape)
                combined_attn_map += avg_attn_map

        # Adjust combined_attn_map size
        target_shape = (img_shape[0] // 32, img_shape[1] // 32)
        combined_attn_map = np.resize(combined_attn_map, target_shape)
        
        combined_attn_map = (combined_attn_map - combined_attn_map.min()) / (combined_attn_map.max() - combined_attn_map.min())
        return combined_attn_map

    def visualize_attention_rollouts(self, all_rollouts, original_frames, title):
        num_frames = len(all_rollouts)
        rows = (num_frames + 7) // 8
        fig, axes = plt.subplots(rows, 8, figsize=(20, 5 * rows))
        axes = axes.flatten()
        
        for i in range(num_frames):
            frame_attention = all_rollouts[i]
            original_frame = original_frames[0,i].squeeze().cpu().numpy().transpose(1, 2, 0)
            original_frame = (original_frame - original_frame.min()) / (original_frame.max() - original_frame.min())
            
            ax = axes[i]
            ax.imshow(original_frame)
            
            # Adjust attention map fit with original frame
            frame_attention_resized = np.array(Image.fromarray((frame_attention * 255).astype(np.uint8)).resize((224, 224), Image.BICUBIC))
            frame_attention_resized = frame_attention_resized.astype(np.float32) / 255.0
            
            ax.imshow(frame_attention_resized, cmap='jet', alpha=0.5, vmin=0, vmax=1)
            ax.set_title(f'Frame {i}')
            ax.axis('off')
        
        for j in range(num_frames, len(axes)):
            axes[j].axis('off')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.savefig('/home/ubuntu/tim/train/result/dual/Attention Rollout.png')
        #plt.show()



    def visualize_3d_grad_cam(self, input_tensor, grad_cam_3d_result):
        num_frames = grad_cam_3d_result.shape[0]
        num_cols = 8  
        num_rows = (num_frames + num_cols - 1) // num_cols  

        plt.figure(figsize=(20, 4 * num_rows))

        for i in range(num_frames):
            plt.subplot(num_rows, num_cols, i + 1)
            
            # Get original frame
            input_frame = input_tensor.squeeze()[i].cpu().permute(1, 2, 0).detach().numpy()
            input_frame = (input_frame - input_frame.min()) / (input_frame.max() - input_frame.min())
            
            
            grad_cam_frame = grad_cam_3d_result[i].cpu().detach().numpy()
            
            
            plt.imshow(input_frame)
            plt.imshow(grad_cam_frame, cmap='jet', alpha=0.8)
            plt.title(f'Frame {i}')
            plt.axis('off')

        plt.tight_layout()
        plt.suptitle('Dual Encoder 3D Grad Cam Map', fontsize=16)
        #plt.show()
        plt.savefig('/home/ubuntu/tim/train/result/dual/3d grad cam.png')
        #plt.close()

    


def main():
    model_path = './model_Swin-Transformer-base1_0803_dual.pth'
    video_path = '/home/ubuntu/tim/train/dataset/train/good/42/part_2/folder_4'
    target_class = 1
    
    vis_tool = VisualizationTool(model_path)
    input_tensor = vis_tool.preprocess_video(video_path)
    
    with torch.no_grad():
        output = vis_tool.model(input_tensor)
    print("Model output shape:", output.shape)
    print("Model output:", output)
    probabilities = torch.softmax(output, dim=1)
    probability_class_1 = probabilities[0, 1].item()
    print("Softmax probabilities:", probabilities)
    
    threshold = 0.5  
    predicted_class = 1 if probability_class_1 > threshold else 0
    

    grad_cam_3d_result = vis_tool.grad_cam_3d(input_tensor, target_class)
    print(grad_cam_3d_result)
    vis_tool.plot_frame_importance(grad_cam_3d_result)
    vis_tool.visualize_3d_grad_cam(input_tensor, grad_cam_3d_result)
    
    feature_maps = vis_tool.get_all_attention_maps(input_tensor)
    
    num_frames = len(feature_maps)  
    all_rollouts = vis_tool.compute_attention_rollout(feature_maps)
    
    vis_tool.compute_and_visualize_attention_rollout(feature_maps, input_tensor)

    print(f"Probability of class 1: {probability_class_1:.4f}")
    print(f"Predicted class: {predicted_class}")

if __name__ == "__main__":
    main()


