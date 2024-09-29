# Import torch for creating and training the model (i.e, optimizer, loss, conv layers)
import torch

# Import Dataset class to pair image and labels and Dataloader to load data in batches for training
from torch.utils.data import Dataset, DataLoader

# Import numpy to load the labels
import numpy as np

# Import tqdm for a more appealing visual process
from tqdm import tqdm

# Import torch.nn to define the loss
import torch.nn as nn

# Import torch optim to define optimizer
import torch.optim as optim

# Import model
from Prob2 import MyModel

# Import matplot to make a plot for the loss
import matplotlib.pyplot as plt

# Import PIL to load images
from PIL import Image

# Import torch transformations to manipulate the images
import torchvision.transforms as T

# Import os to get image file names
import os

# Define the device to use
device = "cuda" if torch.cuda.is_available() else "cpu"

# Set batch size for training dataloaders
batch_size = 128

# #######################################################################################################################
# ####    EXAMPLE ON LOADING THE DATA FOR TRAINING

# ###
# ###   Data must be image list (not path but list of arrays), hour list and minute
# ###   Below is the example but files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0])) might need to be modified
# ###   Once you have image list, hour list and minute list use My_dataset as follows
# ###   train_data = My_Dataset(img_list, hour_labels, minute_labels, transform=my_transform)

# ###  Define paths

# # Full path to the folder that contains the train dataset
# full_path_to_folder = "/home/xiyana/Documents/clock/version_1/"
# # Name of train dataset folder
# path = "train_images/"
# # Name of train labels file
# lab_folder_path = "train_labels.npy"
# # Combine folder name to datasets with the path to train data
# img_folder_path = full_path_to_folder + path

# ### Define function to load images and obtain an ordered list
# ### Make sure images correspond to labels by ordering indexes


# # Function to get the train and/or validation datasets
# def get_training_images(path, img_folder_path):
#     # Get list of files from the folder path
#     files = os.listdir(img_folder_path)

#     # Create a list for the images
#     img_list = []

#     # Sort the images to be in order for training -> This depends on how the files are named
#     # In this case it is for a name such as: clock_1, clock_2, clock_3 ...
#     files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

#     # For every file, we first open the image and then get the dataset using dataset class
#     for file in tqdm(files):
#         # Combine path to folder of the train or validation with the file name
#         the_path = path + file
#         # Open the image using PIL because is more optimal than cv2 with Torch transformations
#         images = Image.open(the_path)
#         # Append image after tranformation to the list
#         img_list.append(images)

#     return img_list


# ### Obtain the image list, hour list and mintue list

# # Load the hour and minute labels for the images
# labeles = np.load(lab_folder_path)

# # Obtain hour labels (0~11 where 0 is 12)
# hour_labels = labeles[:, 0]

# # Obatin minute labels (0~59)
# minute_labels = labeles[:, 1]

# # Obtain image list
# img_list = get_training_images(path, img_folder_path)

# ####### END OF EXAMPLE
# ######################################################################################################################


# Class data for the dataset
class My_Dataset(Dataset):
    def __init__(self, target_img, target_hour, target_minute, transform=None):
        # Init image, hour and minute variable (array for image)
        self.target_img = target_img
        self.target_hour = target_hour
        self.target_minute = target_minute
        self.transform = transform
        # Define lenght of dataset
        self.num_data = len(target_img)

    def __len__(self):
        # Set lenght of dataset
        return self.num_data

    def __getitem__(self, idx):
        # Create a dictionary and store the data
        sample = dict()

        image = self.target_img[idx]

        if self.transform:
            image = self.transform(image)

        sample["img"] = image
        sample["hour"] = self.target_hour[idx]
        sample["minute"] = self.target_minute[idx]

        return sample


# Define the transformation for the image
my_transform = T.Compose(
    [  # Convert the image array to tensor
        T.ToTensor(),
        # Center crop the image
        T.CenterCrop((64, 64)),
    ]
)


# Obtain the training paired data
train_data = My_Dataset(img_list, hour_labels, minute_labels, transform=my_transform)

# Obtain the batches
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=0)

# Define the model
model = MyModel(3, 12, 60)

# Load model to preferred device
model = model.to(device)

# Define the optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Define loss criterion
criterion = nn.CrossEntropyLoss()

# Define max number of epochs
max_epoch = 98

# Define error array
train_errors = []


# Train model for max_epoch number of epochs
for epoch in tqdm(range(max_epoch)):
    # set model to train
    model.train()
    # Define train loss
    train_loss = 0.0

    # Train model for the number of batches in the loader
    for idx, sample in tqdm(enumerate(train_loader)):
        # Set gradients to zero to prevent the accumulation of gradients
        optimizer.zero_grad()

        # Load the image and labels to the device
        img = sample["img"].to(device)
        h_label = sample["hour"].to(device)
        m_label = sample["minute"].to(device)

        # Predict the hour and minute from input image using the model
        pred_h, pred_m = model(img)

        # Compute losses for hour and minute labels
        pred_loss_h = criterion(pred_h, h_label)
        pred_loss_m = criterion(pred_m, m_label)

        # Add losses
        pred_loss = pred_loss_h + pred_loss_m

        # Propagate back losses
        pred_loss.backward()

        # Do an optimizer step to update the parameters
        optimizer.step()

        # Add iteration losses
        train_loss += pred_loss

    # Obtain the average epoch loss
    train_loss = train_loss / len(train_loader)

    # Print the average epoch loss
    print("Avg epoch loss= ", train_loss.item())

    # Save the model
    torch.save(model.state_dict(), "Prob3.pth")

    # Append the error to the error list
    train_errors.append(train_loss.item())


# Define a function to plot training error
def plot_errors(train_errors):
    # Obtain number epochs
    epochs = range(1, len(train_errors) + 1)
    # Plot training errors
    plt.plot(epochs, train_errors, "b", label="Training Error")
    # Add a tittle
    plt.title("Training Error")
    # Add x label
    plt.xlabel("Epochs")
    # Add y label
    plt.ylabel("Error")
    # Add the legend
    plt.legend()
    # Show the plot
    plt.show()


# Plot the error
plot_errors(train_errors)
