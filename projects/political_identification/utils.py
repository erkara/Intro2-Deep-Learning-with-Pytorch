import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Audio, display
from PIL import Image



# ========pytorch modules=======#
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import torch.nn.functional as F
# =======================#
import time
import shutil
import copy
import os
import random
import sys
from datetime import datetime
import warnings
from sklearn.metrics import confusion_matrix
warnings.filterwarnings('ignore')

import librosa
import librosa
import librosa.display
import IPython

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name(0))


# this will make your code repetable to a great extend
def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True



def get_weights(train_set):
    targets = train_set.tensors[1]
    class_counts = torch.bincount(targets)
    class_weights = 100 * (1.0 / class_counts.float())
    sample_weights = class_weights[targets]

    return class_weights, sample_weights





def create_train_test(data_dir, test_ratio=0.2):
    # create the class list
    class_list = []
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")
    for (dirpath, dirnames, filenames) in os.walk(data_dir):
        if len(dirnames) != 0:
            class_list = dirnames

    # remove if exist or create new train/test folders.
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    else:
        os.makedirs(train_dir)

    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    else:
        os.makedirs(test_dir)

    # now create class folders under train/test
    try:
        for cls in class_list:
            os.makedirs(os.path.join(train_dir, cls))
            os.makedirs(os.path.join(test_dir, cls))
    except OSError:
        pass

    for cls in class_list:
        source = os.path.join(data_dir, cls)

        # get all filenames in class-i
        allFileNames = os.listdir(source)
        np.random.shuffle(allFileNames)

        # make the split of filanames

        train_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                                   [int(len(allFileNames) * (1 - test_ratio))])
        # path-list for train and testing filenames
        train_FileNames = [os.path.join(source, name) for name in train_FileNames.tolist()]
        test_FileNames = [os.path.join(source, name) for name in test_FileNames.tolist()]

        # copy the files in /train and /test directories
        for name in train_FileNames:
            shutil.copy(name, os.path.join(train_dir, cls))
        for name in test_FileNames:
            shutil.copy(name, os.path.join(test_dir, cls))

    print(f'train-test split done!\ncheck {data_dir + "/train"} and {data_dir + "/test"}')


# let's look at some images on nxn grid
def display_audio_grid(df, n):
    # select n x n random samples from the dataframe
    df_sample = df.sample(n=n, replace=False)

    # iterate over the samples and load each audio file
    audio_data = []
    for i, row in df_sample.iterrows():
        # load the audio file using librosa
        audio, sr = librosa.load(row['path'], sr=None)
        audio_data.append(audio)

    # create an Audio object for each signal and display a clickable button
    for i, audio in enumerate(audio_data):
        title = df_sample.iloc[i]['label']
        display(f'{title}')
        display(Audio(audio, rate=sr, autoplay=False, embed=True))


def view_class_dist(data):
    fig, ax = plt.subplots()
    counts = dict(data['label'].value_counts())
    class_list = counts.keys()
    class_counts = counts.values()
    ax.bar(class_list, class_counts, width=0.5, align='center')
    plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
    plt.tight_layout()
    plt.show()


def load_model(model, model_path, device=torch.device('cpu')):
    # add optimizer for retraining.
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.best_scores = checkpoint['hyperparams']
    return model


def run_model(model, data_loader, optimizer=None, class_weights=None, mode='train'):
    # Check if the mode is valid ('train' or 'test')
    assert mode in ['train', 'test'], "Invalid mode, choose either 'train' or 'test'"

    # Set the model to training mode if mode is 'train', otherwise set it to evaluation mode
    if mode == 'train':
        model.train()
    else:
        model.eval()

    # Send the model to the device (GPU or CPU)
    model.to(device)

    # Define the loss function (CrossEntropyLoss for multi-class classification)
    if class_weights is not None and mode == 'train':
        weight = class_weights.to(device)
    else:
        weight = None

    # Initialize variables to store total loss and total correct predictions
    total_loss = 0.0
    total_correct = 0

    # Iterate through the data loader (either train or test)
    for i, (images, labels) in enumerate(data_loader):
        # Clear the CUDA cache to free up memory
        torch.cuda.empty_cache()

        # Send images and labels to the device (GPU or CPU)
        images = images.to(device)
        labels = labels.to(device)

        # Compute the model output (predictions) for the current batch of images
        outputs = model(images)

        # Calculate the loss using the criterion defined above
        # loss = criterion(outputs, labels)
        loss = F.cross_entropy(outputs, labels, weight=weight)

        # If in training mode, perform backpropagation and update model weights
        if mode == 'train':
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Accumulate the loss for the current batch
        total_loss += loss.item() * images.size(0)

        # Get the predicted class for each image in the batch
        predictions = outputs.argmax(dim=1)

        # Compare the predicted class with the true label, and count the correct predictions
        correct_counts = predictions.eq(labels)

        # Accumulate the total correct predictions for all batches
        total_correct += correct_counts.sum().item()

    # Calculate the average loss and accuracy for the entire dataset
    avg_loss = total_loss / len(data_loader.dataset)
    accuracy = total_correct / len(data_loader.dataset)

    # Return accuracy and average loss
    return accuracy, avg_loss


def train_and_test(model, train_loader, test_loader, optimizer, num_epochs=1, log_int=1,
                   class_weights=None, save_best=False, save_dir='./runs'):
    # Create a new directory for the current run
    #current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    #run_dir = os.path.join(save_dir, current_time)
    run_dir = save_dir
    if save_best:
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)

    # Lists to store losses and accuracies for each epoch
    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    # Initialize the best test accuracy to 0.0
    best_test_accuracy = 0.0

    # Get the initial learning rate and weight decay values
    init_lr = optimizer.param_groups[0]['lr']
    init_wd = optimizer.param_groups[0]['weight_decay']

    params = {
        'batch_size': train_loader.batch_size,
        'learning_rate': init_lr,
        'weight_decay': init_wd,
        'optimizer': type(optimizer).__name__
    }

    # Iterate through each epoch
    for epoch in range(num_epochs):

        # Train the model using the 'run_model()' function in 'train' mode
        train_acc, train_loss = run_model(model, train_loader, optimizer, class_weights, mode='train', )
        # Append the training loss and accuracy for the current epoch
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Test the model using the 'run_model()' function in 'test' mode
        test_acc, test_loss = run_model(model, test_loader, mode='test')
        # Append the test loss and accuracy for the current epoch
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        # Print the train and test accuracy for the current epoch
        if (epoch + 1) % log_int == 0:
            print(
                f"epoch: {epoch + 1}/{num_epochs} train_acc: {train_acc * 100:.2f}%,test_acc: {test_acc * 100:.2f}%")

        # Check if the current test accuracy is greater than the best test accuracy
        if test_acc > best_test_accuracy:
            print(
                f"improved! epoch: {epoch + 1}/{num_epochs} train_acc: {train_acc * 100:.2f}%, test_acc: {test_acc * 100:.2f}%")
            # Update the best test accuracy
            best_test_accuracy = test_acc
            params['best_acc'] = round(best_test_accuracy, 3)
            if save_best:
                torch.save({'hyperparams': params,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                           f'{run_dir}/best_model.pth')

    print(f"best testing accuracy: {100 * best_test_accuracy:0.1f}")
    return round(best_test_accuracy, 3)


def get_confusion_matrix(model, test_loader):
    with torch.no_grad():
        model.eval()
        model.to(device)
        all_preds = torch.FloatTensor([]).to(device)
        all_labels = torch.LongTensor([]).to(device)
        # compute all predictions
        for images, labels in iter(test_loader):
            images, labels = images.to(device), labels.to(device)
            preds = model(images)
            all_preds = torch.cat(
                (all_preds, preds),
                dim=0
            )
            all_labels = torch.cat((all_labels, labels), dim=0)

        # get the predicted labels
        preds = all_preds.argmax(dim=1)
        # create confusion matrix
        cm = confusion_matrix(all_labels.cpu().numpy(), preds.cpu().numpy())
        return cm


def plot_confusion_matrix(cm, class_list):
    # display confusion matrix. truth:x_axis(rows), preds:y_axis(cols)
    df_cm = pd.DataFrame(cm, index=[i for i in class_list], columns=[i for i in class_list])
    plt.figure(figsize=(12, 6))
    ax = sns.heatmap(df_cm, annot=True, cmap="YlGnBu", fmt='g', annot_kws={'fontsize': 15})
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=15)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=15)
    plt.xlabel('predictions', fontsize=20)
    plt.ylabel('ground_truth')
    plt.show()


# returns major matrics
def get_scores(cm, score_type):
    accuracy = np.around(np.sum(np.diag(cm)) / np.sum(cm), 2)
    recalls = np.around(np.diag(cm) / np.sum(cm, axis=1), 2)
    precisions = np.around(np.diag(cm) / np.sum(cm, axis=0), 2)
    f1 = np.around(2 / (1 / recalls + 1 / precisions), 2)
    if score_type == 'accuracy':
        return accuracy
    if score_type == 'recall':
        return recalls
    if score_type == 'precision':
        return precisions
    if score_type == 'F1':
        return f1
    else:
        print('Enter one of the score types \n accuracy,recalls,precisions or F1 ')


def display_all_scores(cm, class_list):
    score_list = ['recall', 'precision', 'F1']
    fig, ax = plt.subplots(1, 3, figsize=(18, 9), sharey=True)
    for i, score_type in enumerate(score_list):
        scores = get_scores(cm, score_type=score_type)
        ax[i].bar(class_list, scores, color='orange', width=0.5, align='center')
        ax[i].set_xticklabels(class_list, rotation=0, fontsize=20)
        title = 'mean-' + score_type + ':' + str(np.round(scores.mean(), 2))
        ax[i].set_title(title, color='magenta', fontsize=25)
        # Add labels with scores above each bar
        for j, score in enumerate(scores):
            ax[i].text(j, score + 0.01, f"{score:.2f}", ha='center', fontsize=25, color='magenta')
        ax[i].set_ylim([0, 1.1])

    plt.tight_layout()
    plt.show()
