import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#use standard sizes for all figures
SMALL_SIZE = 18
MEDIUM_SIZE = 22
BIGGER_SIZE = 30


plt.rc('font', size = BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize = BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize = BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize = SMALL_SIZE)   # fontsize of the tick labels
plt.rc('ytick', labelsize = SMALL_SIZE)   # fontsize of the tick labels
plt.rc('legend', fontsize = BIGGER_SIZE)   # legend fontsize
plt.rc('figure', titlesize = BIGGER_SIZE)  # fontsize of the figure title




from PIL import Image
import seaborn as sns
#========pytorch modules=======#
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import torch.nn.functional as F
#=======================#
import time
import shutil
import copy
import os
import random
import sys
import warnings
from sklearn.metrics import confusion_matrix
warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name(0))
#this will make your code repetable to a great extend





def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True




def create_train_test(data_dir, test_ratio = 0.2):
    #create the class list
    class_list = []
    train_dir = os.path.join(data_dir,"train")
    test_dir = os.path.join(data_dir,"test")
    for (dirpath, dirnames, filenames) in os.walk(data_dir):
        if len(dirnames)!=0:
            class_list = dirnames
            
    #remove if exist or create new train/test folders.
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    else:
        os.makedirs(train_dir)

    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    else:
        os.makedirs(test_dir)

    #now create class folders under train/test
    try:
        for cls in class_list:
            os.makedirs(os.path.join(train_dir,cls))
            os.makedirs(os.path.join(test_dir,cls))
    except OSError:
        pass
    
    for cls in class_list:
        source = os.path.join(data_dir,cls)

        #get all filenames in class-i
        allFileNames = os.listdir(source)
        np.random.shuffle(allFileNames)

        #make the split of filanames
        
        train_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                                          [int(len(allFileNames)* (1 - test_ratio))])
        #path-list for train and testing filenames
        train_FileNames = [os.path.join(source,name) for name in train_FileNames.tolist()]
        test_FileNames = [os.path.join(source,name) for name in test_FileNames.tolist()]

        #copy the files in /train and /test directories 
        for name in train_FileNames:
            shutil.copy(name, os.path.join(train_dir,cls))
        for name in test_FileNames:
            shutil.copy(name, os.path.join(test_dir,cls))
    
    print(f'train-test split done!\ncheck {data_dir +"/train"} and {data_dir +"/test"}')






def resize_save(data_dir, save_dir, dim=(224, 224), RGB=True):
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    else:
        os.makedirs(save_dir)

    for (dirpath, dirnames, filenames) in os.walk(data_dir):
        new_paths = dirpath.replace(data_dir, save_dir)
        for dirname in dirnames:
            cls_dir = os.path.join(save_dir, dirname)
            if os.path.exists(cls_dir):
                shutil.rmtree(cls_dir)
            else:
                os.makedirs(cls_dir)
        for names in filenames:
            img_path = os.path.join(dirpath, names)
            if RGB:
                img = Image.open(img_path).convert("RGB")

            else:
                img = Image.open(img_path).convert("L")

            img = img.resize(dim, Image.NEAREST)

            new_dir = dirpath.replace(data_dir, save_dir)
            img.save(os.path.join(new_dir, names))

    print('resizing done!')


# get all train and test image paths to be use later
def get_data_paths(train_dir, test_dir):
    train_image_paths = []
    test_image_paths = []
    for (dirpath, dirnames, filenames) in os.walk(train_dir):
        for names in filenames:
            train_image_paths.append(os.path.join(dirpath, names))
    for (dirpath, dirnames, filenames) in os.walk(test_dir):
        for names in filenames:
            test_image_paths.append(os.path.join(dirpath, names))

    random.shuffle(train_image_paths)
    random.shuffle(test_image_paths)

    return train_image_paths, test_image_paths


# let's look at some images on nxn grid
def view_samples(n, img_paths):
    plt.figure(figsize=(12, 12))
    img_lst = random.sample(img_paths, n * n)
    for i, img_path in enumerate(img_lst):
        img = Image.open(img_path).convert("RGB")
        label = img_path.split('/')[-2]
        plt.subplot(n, n, i + 1)
        plt.subplots_adjust(wspace=0.01, hspace=0.001)
        plt.title(label, fontsize=20)
        plt.axis('off')
        plt.imshow(img)
    plt.tight_layout()
    plt.show()


def view_class_dist(class_list, class_counts):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(class_list, class_counts, width=0.5, align='center')
    plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
    plt.tight_layout()
    plt.show()





def load_model(model, model_path,device=torch.device('cpu')):
    # add optimizer for retraining.
    checkpoint = torch.load(model_path,map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.best_scores = checkpoint['best_stats']
    return model



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

    # Initialize the loss function (CrossEntropyLoss for multi-class classification)
    criterion = nn.CrossEntropyLoss()

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
        loss = criterion(outputs, labels)

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

def train_test_model(model, train_loader, test_loader, optimizer, num_epochs=1,log_int=1,model_name='best_model'):
    # Lists to store losses and accuracies for each epoch
    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    # Initialize the best test accuracy to 0.0
    best_test_accuracy = 0.0
    
    #train time
    start_time=time.time()
    
    # Iterate through each epoch
    for epoch in range(num_epochs):
        # Train the model using the 'run_model()' function in 'train' mode
        train_acc, train_loss = run_model(model, train_loader, optimizer, mode='train')
        # Append the training loss and accuracy for the current epoch
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # Test the model using the 'run_model()' function in 'test' mode
        test_acc, test_loss = run_model(model, test_loader, mode='test')
        # Append the test loss and accuracy for the current epoch
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        
        # Print the train and test accuracy for the current epoch
        if (epoch+1)%log_int==0:
            print(f"epoch:{epoch + 1}/{num_epochs} train_acc: {train_acc * 100:.1f}%, testing_acc: {test_acc * 100:.1f}%")

        # Check if the current test accuracy is greater than the best test accuracy
        if test_acc > best_test_accuracy:
            # Update the best test accuracy
            best_test_accuracy = test_acc
            
            # Save the model's state dictionary with some stats
            best_state_dic = copy.deepcopy(model.state_dict())
            beststats = {'BestTestAcc': best_test_accuracy,
                         'lr': optimizer.param_groups[0]['lr'],
                         }
            checkpoint = {'state_dict': best_state_dic,
                          'best_stats': beststats
                          }
            #save the model
            torch.save(checkpoint, f"{model_name}.pth")
            
            
    end_time  = time.time()
    print(f"best testing accuracy: {100*best_test_accuracy:.2f} train time: {end_time-start_time:0.1f}sec")






def get_class_probs(model, image_path, test_transforms,idx_to_class,topk=3):
    with torch.no_grad():
        model.to(device)
        model.eval()
        img = Image.open(image_path).convert("RGB")
        img = test_transforms(img)
        img = img.unsqueeze(0).to(device)
        output = model(img)
        # remember softmax
        probs = F.softmax(output)
        top_probabilities, top_indices = probs.topk(topk)
        top_probabilities = top_probabilities.detach().type(torch.FloatTensor).numpy().tolist()[0]
        top_probabilities = [round(elem, 2) for elem in top_probabilities]
        top_indices = top_indices.detach().type(torch.FloatTensor).numpy().tolist()[0]
        top_classes = [idx_to_class[index] for index in top_indices]

    return top_probabilities, top_classes





def get_misclassified_image_paths(model, test_image_paths,test_transforms, idx_to_class,topk):
    if topk>len(idx_to_class):
        print(f"Entered topk = {topk}>num_classes, switching to topk = {len(idx_to_class)} ")
        topk = len(idx_to_class)

    misclass_image_paths = []
    topk_image_paths = []
    for img_path in test_image_paths:
        top_probabilities, top_classes = get_class_probs(model, img_path, test_transforms,idx_to_class,topk)
        label = img_path.split('/')[-2]
        if label != top_classes[0]:
            misclass_image_paths.append(img_path)
            if label in top_classes[0:topk]:
                topk_image_paths.append(img_path)

    topk_accuracy = 1 - (len(misclass_image_paths) - len(topk_image_paths)) / len(test_image_paths)
    print(f"top-{topk} accuracy: {100*topk_accuracy:0.2f}%")

    return misclass_image_paths, topk_image_paths


def display_misclassified_images(model,test_image_paths,n,topk,test_transforms,idx_to_class):
    if topk>len(idx_to_class):
        print(f"Entered topk = {topk}>num_classes, switching to topk = {len(idx_to_class)} ")
        topk = len(idx_to_class)

    misclass_image_paths = []
    topk_image_paths = []
    for img_path in test_image_paths:
        top_probabilities, top_classes = get_class_probs(model, img_path, test_transforms,idx_to_class,2)
        label = img_path.split('/')[-2]
        if label != top_classes[0]:
            misclass_image_paths.append(img_path)
            if label in top_classes[0:2]:
                topk_image_paths.append(img_path)

    top2_accuracy = 1 - (len(misclass_image_paths) - len(topk_image_paths)) / len(test_image_paths)
    print(f"top2_accuracy: {100*top2_accuracy:0.2f}%")
    display_predictions(model,misclass_image_paths,n,topk,test_transforms,idx_to_class)

    
	
	

def display_predictions(model,img_paths, n,topk,test_transforms,idx_to_class):
    num_class = len(idx_to_class)
    if topk>num_class:
        topk = num_class

    plt.subplots(n, 2, figsize=(12, 4 * n))
    plt.subplots_adjust(0, 0, 3, 3, wspace=0.75)
    rnd_number = random.sample(range(0, len(img_paths)-1), n)
    for i in range(1, n + 1):
        img_path = img_paths[rnd_number[i-1]]
        label = img_path.split('/')[-2]
        image = Image.open(img_path).convert("RGB")
        plt.subplot(n, 2, 2 * i - 1)
        plt.axis('off')
        plt.title(label)
        plt.imshow(image)
        probs, classes = get_class_probs(model, img_path, test_transforms,idx_to_class,topk=topk)
        plt.subplot(n, 2, 2 * i)
        plt.barh(classes[::-1], probs[::-1], align='center', color='orange')
        for index, value in enumerate(probs[::-1]):
            plt.text(value, index, str(value), color='red', fontsize=20)
        plt.tight_layout()

    plt.show()



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


def plot_confusion_matrix(cm,class_list):
    #display confusion matrix. truth:x_axis(rows), preds:y_axis(cols)
    df_cm = pd.DataFrame(cm, index = [i for i in class_list], columns = [i for i in class_list])
    plt.figure(figsize=(18,8))
    sns.heatmap(df_cm,annot=True,cmap="YlGnBu",fmt='g',annot_kws={'fontsize': 16})
    plt.xlabel('predictions')
    plt.ylabel('ground truth')
    plt.show()




#returns major matrics
def get_scores(cm,score_type):
    accuracy = np.around(np.sum(np.diag(cm))/np.sum(cm),2)
    recalls = np.around(np.diag(cm)/np.sum(cm,axis = 1),2)
    precisions = np.around(np.diag(cm)/np.sum(cm,axis = 0),2)
    f1 = np.around( 2/(1/recalls + 1/precisions),2)
    if score_type=='accuracy':
        return accuracy
    if score_type=='recall':
        return recalls
    if score_type=='precision':
        return precisions
    if score_type=='F1':
        return f1
    else:
        print('Enter one of the score types \n accuracy,recalls,precisions or F1 ')




def display_all_scores(cm,class_list):
    score_list = ['recall','precision','F1']
    fig, ax = plt.subplots(1,3,figsize=(18,8),sharey=True)
    for i,score_type in enumerate(score_list):
        scores = get_scores(cm,score_type=score_type)
        ax[i].bar(class_list, scores,color='orange',width=0.5, align='center')
        ax[i].set_xticklabels(class_list, rotation=75)
        title = 'mean_' + score_type +':'+ str(np.round(scores.mean(),2))
        ax[i].set_title(title,color='magenta')
        # Add labels with scores above each bar
        for j, score in enumerate(scores):
            ax[i].text(j, score + 0.01, f"{score:.2f}", ha='center', fontsize=20,color='magenta')
        ax[i].set_ylim([0, 1])

    plt.tight_layout()
    plt.show()








visualisation = {}
def hook_fn(m, i, o):
    visualisation[m] = o

def get_all_feature_maps(model, image_path, test_transforms,min_image_size=14):
    layer_list = []
    selected_layers = []
    vis_dict = {}
    with torch.no_grad():
        model.to(device)
        model.eval()
        img = Image.open(image_path).convert("RGB")
        label = image_path.split('/')[-2]
        img = test_transforms(img)
        img = img.unsqueeze(0).to(device)
        for layer in model.modules():
            if isinstance(layer, torch.nn.modules.conv.Conv2d):
                layer.register_forward_hook(hook_fn)
                layer_list.append(layer)

        output = model(img)

    for i, j in enumerate(layer_list):
        layer_name = layer_list[i]
        out = visualisation[layer_name]
        if out.shape[3] >= min_image_size:
            selected_layers.append(layer_name)
            vis_dict[i] = out
    print(f'{len(vis_dict)}/{len(layer_list)} convolutional layers is selected')
    print(f"pick one of these layer numbers to visualize {vis_dict.keys()}")
    return vis_dict, selected_layers


def visualize_feature_maps(vis_dict,image_path,layer_num=0):
    #plot the original image
    img = Image.open(image_path).convert('L')
    plt.imshow(img,aspect='auto',cmap="gray")
    plt.axis('off')
    plt.title('orginal image')
    plt.show()
    #get all feature maps
    output = vis_dict[layer_num]
    print(f'feature map size: {output.shape[2]}x{output.shape[2]}')
    num_feature_maps = output[0].shape[0]
    if num_feature_maps<=64:
        s = int(np.ceil(np.sqrt(num_feature_maps)))
    else:
        print(f'I will display 64/{num_feature_maps} feature maps!')
        s = 8
    plt.figure(figsize=(20, 15))
    for i, image in enumerate(output[0]):
        if i==64:
            break
        image = image.squeeze(0)
        plt.subplot(s,s,i+1)
        plt.imshow(image.cpu().numpy(),cmap='gray')
        plt.title(f'feature{i+1}',fontsize=15)
        plt.axis('off')
    plt.tight_layout()
    plt.show()




