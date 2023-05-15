import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import torch
from torchvision import models
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torch.nn.functional as F
from torchinfo import summary
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split
import albumentations as A  
from albumentations.pytorch import ToTensorV2
import os
import copy
import gdown
import zipfile

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import warnings
warnings.filterwarnings('ignore') 


def download_data():
    file_id = '1D9pzK8CbpH5y08sU1trqosgypOPK680H'
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'output.zip'  # Specify the desired output file name and path
    gdown.download(url, output, quiet=False)

    zip_file = 'output.zip'

    # Extract the contents of the zip file
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall()

    # Delete the zip file
    os.remove(zip_file)



#this will create the backbone dataframe
def get_image_mask_paths(root_dir):
    # Get all class folder names, ignoring any files in root_dir
    class_folders = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

    data = {
        'class_name': [],
        'image_path': [],
        'mask_path': []
    }

    # Iterate through each class folder
    for class_folder in class_folders:
        class_dir = os.path.join(root_dir, class_folder)

        # List all files in the class directory
        all_files = os.listdir(class_dir)

        # Separate image and mask files
        image_files = sorted([f for f in all_files if not f.endswith('_mask.tif')])

        # Append data for the current class folder
        for img_file in image_files:
            img_name = os.path.splitext(img_file)[0]
            mask_file = f'{img_name}_mask.tif'

            if mask_file in all_files:
                data['class_name'].append(img_name)
                data['image_path'].append(os.path.join(class_dir, img_file))
                data['mask_path'].append(os.path.join(class_dir, mask_file))

    # Create a DataFrame
    df = pd.DataFrame(data)
    
    return df


#return samples with or without tumor in it.
def get_negative_positive_samples(df):
    # Function to check if a mask is positive (has at least one white pixel)
    def is_positive_mask(mask_path):
        mask = np.array(Image.open(mask_path).convert("L"))
        return np.any(mask > 0)

    # Filter the dataframe to keep only rows with positive masks
    positive_df = df[df['mask_path'].apply(is_positive_mask)]

    # Reset the index of the new dataframe
    positive_df = positive_df.reset_index(drop=True)

    # Filter the dataframe to keep only rows with negative masks
    negative_df = df[~df['mask_path'].apply(is_positive_mask)]
    
    # Reset the index of the new dataframe
    negative_df = negative_df.reset_index(drop=True)

    return positive_df, negative_df


def display_image_mask_pairs(dataframe, num_pairs=2):
    fig, axes = plt.subplots(nrows=3, ncols=num_pairs, figsize=(4 * num_pairs, 12))

    # Get a random subset of samples
    random_samples = dataframe.sample(n=num_pairs)

    for i, (_, row) in enumerate(random_samples.iterrows()):
        # Read image and mask
        img_path = row['image_path']
        mask_path = row['mask_path']
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        # Create red mask
        mask_np = np.array(mask)
        red_mask = np.zeros_like(img)
        red_mask[..., 0] = mask_np

        # Overlay image and mask with opacity
        img_np = np.array(img)
        opacity = 0.7
        overlay = np.where(mask_np[..., None] > 0, (opacity * red_mask + (1 - opacity) * img_np).astype(np.uint8), img_np)

        # Display image, mask, and overlay
        axes[0, i].imshow(img)
        axes[0, i].axis('off')
        axes[0, i].set_title(f'Image {i + 1}',fontsize=20)

        axes[1, i].imshow(mask, cmap='viridis')
        axes[1, i].axis('off')
        axes[1, i].set_title(f'Mask {i + 1}',fontsize=20)

        axes[2, i].imshow(overlay)
        axes[2, i].axis('off')
        axes[2, i].set_title(f'Overlay {i + 1}',fontsize=20)

    plt.tight_layout()
    plt.show()
    
    
    
def display_class_distribution(dataframe):
    positive_count = 0
    negative_count = 0

    for _, row in dataframe.iterrows():
        mask_path = row['mask_path']
        mask = Image.open(mask_path).convert('L')
        mask_np = np.array(mask)

        if mask_np.max() > 0:
            positive_count += 1
        else:
            negative_count += 1

    # Plot class distribution
    plt.bar(['Negative Mask', 'Positive Mask'], [negative_count, positive_count])
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Class Distribution')
    plt.show()
    
    
def iou_pytorch(predictions: torch.Tensor, labels: torch.Tensor, e: float = 1e-7):
    """Calculates Intersection over Union for a tensor of predictions"""
    predictions = torch.where(predictions > 0.5, 1, 0)
    labels = labels.byte()
    
    intersection = (predictions & labels).float().sum((1, 2))
    union = (predictions | labels).float().sum((1, 2))
    
    iou = (intersection + e) / (union + e)
    return iou

def dice_pytorch(predictions: torch.Tensor, labels: torch.Tensor, e: float = 1e-7):
    """Calculates Dice coefficient for a tensor of predictions"""
    predictions = torch.where(predictions > 0.5, 1, 0)
    labels = labels.byte()
    
    intersection = (predictions & labels).float().sum((1, 2))
    return ((2 * intersection) + e) / (predictions.float().sum((1, 2)) + labels.float().sum((1, 2)) + e)


def BCE_dice(output, target, alpha=0.01):
    bce = torch.nn.functional.binary_cross_entropy(output, target)
    soft_dice = 1 - dice_pytorch(output, target).mean()
    return bce + alpha * soft_dice


def run_model(model, data_loader, optimizer=None, mode='train'):
    # Check if the mode is valid ('train' or 'test')
    assert mode in ['train', 'test'], "Invalid mode, choose either 'train' or 'test'"

    # Set the model to training mode if mode is 'train', otherwise set it to evaluation mode
    if mode == 'train':
        model.train()
    else:
        model.eval()

    # Send the model to the device (GPU or CPU)
    model.to(device)

    # Initialize the loss function (BinaryCrossEntropyLoss for binary segmentation)
    criterion = BCE_dice

    # Initialize variables to store total loss and total correct predictions
    total_loss = 0.0
    total_iou = 0.0


    # Initialize variables to store total loss and total correct predictions
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0


    # Iterate through the data loader (either train or test)
    for i, (images, masks) in enumerate(data_loader):

        # Clear the CUDA cache to free up memory
        torch.cuda.empty_cache()

        # Send images and masks to the device (GPU or CPU)
        images = images.to(device).float()  # convert to float
        masks = masks.to(device)

        # Compute the model output (predictions) for the current batch of images
        outputs = model(images)
        outputs = outputs.squeeze(1)

        # Calculate the loss using the criterion defined above
        loss = criterion(outputs, masks)

        # If in training mode, perform backpropagation and update model weights
        if mode=='train':
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Accumulate the loss for the current batch
        total_loss += loss.item() * images.size(0)
        total_iou += iou_pytorch(outputs, masks).sum().item()
        total_dice += dice_pytorch(outputs, masks).sum().item()




    # # Calculate the average loss, accuracy, and IOU for the entire dataset
    avg_loss = total_loss / len(data_loader.dataset)
    avg_iou = total_iou / len(data_loader.dataset)
    avg_dice = total_dice / len(data_loader.dataset)

    #print(f"avs_loss:{avg_loss:0.3f} avg_iou:{avg_iou:0.3f} avg_dice:{avg_dice:0.3f}")

    # Return accuracy, average loss, and IOU
    return  avg_loss, avg_iou, avg_dice



def train_and_test(model, train_loader, test_loader, optimizer, num_epochs=1, log_int=1, model_name='best_model'):
    # Lists to store losses, accuracies, and IOUs for each epoch
    train_losses, test_losses = [], []
    train_ious, test_ious = [], []
    train_dices,test_dices = [], []

    # Initialize the best test IOU to 0.0
    best_test_iou = 0.0
    
    # Train the model for the specified number of epochs
    for epoch in range(num_epochs):
        # Train the model using the 'run_model()' function in 'train' mode
        train_loss, train_iou, train_dice = run_model(model, train_loader, optimizer, mode='train')
        
        # Test the model using the 'run_model()' function in 'test' mode
        test_loss, test_iou,test_dice = run_model(model, test_loader, mode='test')
        
        # Print the train and test metrics for the current epoch
        if (epoch+1) % log_int == 0:
            print(f"epoch: {epoch+1}/{num_epochs} train_loss: {train_loss * 100:.1f} train_iou: {train_iou * 100:.1f}% train_dice: {train_dice * 100:.1f}%"
                  f" test_loss: {test_loss * 100:.1f} test_iou: {test_iou * 100:.1f}%  test_dice: {test_dice * 100:.1f}%")


        # Check if the current test IOU is greater than the best test IOU
        if test_iou > best_test_iou:
            # Update the best test IOU
            best_test_iou = test_iou
            
            # Save the model's state dictionary with some stats
            best_state_dic = copy.deepcopy(model.state_dict())
            beststats = {'BestTestIOU': round(best_test_iou,3),
                         'lr': optimizer.param_groups[0]['lr'],
                         }
            checkpoint = {'state_dict': best_state_dic,
                          'best_stats': beststats
                          }
            # Save the model
            torch.save(checkpoint, f"{model_name}.pth")
    
    # Print the final results
    print(f"best_IOU: {100 * best_test_iou:.3f}")
    
    # Return the loss, accuracy, and IOU lists for plotting
    return best_test_iou


def load_model(model, model_path,device=torch.device('cpu')):
    # add optimizer for retraining.
    checkpoint = torch.load(model_path,map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.best_scores = checkpoint['best_stats']
    return model


def predict_mask(model, image_path,common_augmentations):
    model.eval()
    model.to(device)
    # Load image and mask using PIL
    img = np.array(Image.open(image_path))

    # Apply common_augmentations to both image and mask
    transformed = common_augmentations(image=img)
    img = transformed["image"]
    #mask = transformed["mask"]
    
    

    # Add batch dimension to image tensor
    img = img.unsqueeze(0).float()

    # Make prediction with the model
    model.eval()
    with torch.no_grad():
        output = model(img.to(device))[0][0]

    # Convert output to binary mask
    predicted_mask = torch.where(output > 0.5, 1, 0)


    return predicted_mask.cpu()



def plot_predicted_samples(model,df,n,common_augmentations):
    fig, axs = plt.subplots(n, 3, figsize=(10, n*4))

    # Randomly select n samples from the dataframe
    samples = df.sample(n)

    for i, (_, row) in enumerate(samples.iterrows()):
        image_path = row['image_path']
        mask_path = row['mask_path']
        
        # Get the predicted mask and original mask
        predicted_mask = predict_mask(model, image_path,common_augmentations)
        
        # Load the original image
        original_image = Image.open(image_path)
        original_mask = Image.open(mask_path)

        # Plot original image, original mask, and predicted mask
        axs[i, 0].imshow(original_image)
        axs[i, 0].set_title("Original Image")
        axs[i, 0].axis("off")

        axs[i, 1].imshow(original_mask, cmap="gray")
        axs[i, 1].set_title("Original Mask")
        axs[i, 1].axis("off")

        axs[i, 2].imshow(predicted_mask, cmap="gray")
        axs[i, 2].set_title("Predicted Mask")
        axs[i, 2].axis("off")

    plt.tight_layout()
    plt.show()
