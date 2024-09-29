import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from torch.utils.data import DataLoader, TensorDataset

# some initial imports
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F


from tqdm import tqdm
from cl_dataset import ContinualMNIST as MNIST
from einops import rearrange
from model import Net
from utils import seed_everything, evaluate_task, train_task


### Mine: Create a class to edit the net
class CustomNet(nn.Module):
    ### Mine: define init for custom net
    def __init__(self, net_instance, f2_new_task_output_size):
        super(CustomNet, self).__init__()

        ### Mine: Use the Net given as base model
        self.net = net_instance

        ### Mine: Unfreeze all layers of the original net
        for param in self.net.parameters():
            param.requires_grad = True

        ### Mine: Define a new head for the new task
        self.fc2_new_task_head = nn.Sequential(nn.Linear(self.net.fc1.out_features, f2_new_task_output_size))

        ### Mine: Unfreeze the parameters of the new task head
        for param in self.fc2_new_task_head.parameters():
            param.requires_grad = True

    ### Mine: Define the forward function based on original net
    def forward(self, x):
        ### Mine: Forward pass through the original network
        x = F.relu(self.net.bn1(self.net.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.net.bn2(self.net.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.net.bn3(self.net.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = self.net.conv_drop(x)
        x = rearrange(x, "b c h w -> b (c h w)")
        x = F.relu(self.net.fc1(x))
        x = F.dropout(x, training=self.net.training)

        ### Mine: pass x to the head for new task (fc2_new_task)
        output_fc2_new_task = self.fc2_new_task_head(x)

        ### Mine: pass x to the Original head (fc2) output
        output_fc2 = self.net.fc2(x)

        ### Mine: Concatenate both outputs
        final_output = torch.cat((output_fc2, output_fc2_new_task), dim=1)

        return final_output


# * ----------------------- global setup ------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed_value = 42
seed_everything(seed_value)

# * ----------------------- hyper params and mdoel ------------------------------
model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

NUM_EPOCH = 3
dataset = MNIST()

# * ----------------------- task process ------------------------------
seen_classes_list = []  # Renamed from accumulate

fisher_dict = {}
old_par_dict = {}
ewc_lambda = 10000


### Mine: Obtain fisher and old parameter dictionaries
def get_fisher_dict(task_id, data_loader, criterion, device):
    ### Mine: Set model to train and do zero grad
    model.train()
    optimizer.zero_grad()

    ### Mine: For every batch send loss backward
    loop = tqdm(data_loader, total=len(data_loader), leave=True)
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

    ### Mine: Initialize a dict for the fisher information of current task
    fisher_dict[task_id] = {}

    ### Mine: Initialize a dict for the old parameters of current task
    old_par_dict[task_id] = {}

    ### Mine: Initialize a dict for the fisher information of current task
    for name, param in model.named_parameters():
        ### Mine: adjust name for compatibility with custom_net
        new_name = "net." + name

        ### Mine: store all parameters
        old_par_dict[task_id][new_name] = param.data.clone()

        ### Mine: Use accumulation of gradients to the power of 2 as approximation of fisher information
        fisher_dict[task_id][new_name] = param.grad.data.clone().pow(2)

    return fisher_dict, old_par_dict


### Mine: Function to train based on Elastic Weight consolidation for continual learning
def ewc_training(model, data_loader, optimizer, criterion, num_epochs, device, task_id, fisher_dict, old_par_dict):
    ### Mine: Set model to train
    model.train()

    ### Mine: For each epoch
    for epoch in range(num_epochs):
        ### Mine: Use some parts from the given train function in utils.py
        total_loss = 0.0
        correct = 0
        total = 0

        ### Mine: Use some parts from the given train function in utils.py
        loop = tqdm(data_loader, total=len(data_loader), leave=True)
        for images, labels in loop:
            ### Mine: send images and labels to device
            images, labels = images.to(device), labels.to(device)

            ### Mine: do zero grad
            optimizer.zero_grad()

            ### Mine: get outputs from model
            outputs = model(images)

            ### Mine: get loss for current task using criterion
            loss = criterion(outputs, labels)

            ### Mine: For parameter we get the fisher information and the old parameter
            for name, param in model.named_parameters():
                ### Mine: Skip parameters for fc2 because we just added this to the network

                if not name.startswith("fc2"):
                    ### Mine: Get fisher information from old task for the given parameters
                    fisher_info = fisher_dict[task_id - 1][name]

                    ### Mine: Get parameters from old task
                    old_parameter = old_par_dict[task_id - 1][name]

                    ### Mine: Compute loss based of Elastic weight consolidation
                    loss += (fisher_info * (param - old_parameter).pow(2)).sum() * ewc_lambda

            ### Mine: Do backpropagation
            loss.backward()

            ### Mine: Update parameters
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update tqdm progress bar
            loop.set_description(f"Epoch [{epoch + 1}/{num_epochs}]")
            loop.set_postfix(loss=loss.item(), acc=100 * correct / total)

        avg_loss = total_loss / len(data_loader)
        accuracy = 100 * correct / total
        print(f"Epoch [{epoch + 1}/{num_epochs}] training - Avg Loss: {avg_loss:.6f}, Accuracy: {accuracy:.6f}%")


for task_idx, (x_train_task, t_train_task, x_test_task, t_test_task) in enumerate(dataset.task_data):
    # Update seen_classes_list
    seen_classes = np.unique(t_train_task).tolist()
    seen_classes_list.extend(seen_classes)
    seen_classes_list = list(
        set(seen_classes_list)
    )  # if use the replay method, for preventing ovelapping each classes

    ### Mine: If seen classes is only first 5 classes (0,1,2,3,4)
    if task_idx == 0:
        print("\n")
        print("Network: NET")

        ### Mine: Simple adapt last layer using the given functions
        model.adapt_last_layer(len(seen_classes_list))

    ### Mine: If seen classes is more than first 5 classes (0,1,2,3,4)
    else:
        print("\n")
        print("Network: CUSTOM NET")

        ### Mine: Use customnet
        model = CustomNet(model, len(seen_classes_list))
        model.to(device)

    # Convert numpy arrays to PyTorch tensors and ensure they're on the correct device
    x_train_task = torch.tensor(x_train_task).float()
    t_train_task = torch.tensor(t_train_task).long()

    train_dataset = TensorDataset(x_train_task, t_train_task)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # * ----------------------- train ------------------------------
    ### Mine: If seen classes is only first 5 classes (0,1,2,3,4)
    if task_idx == 0:
        print("Training style: Normal training")

        ### Mine: Train using normal training
        train_task(model, train_loader, optimizer, criterion, NUM_EPOCH)

        ### Mine: Get fisher information and old parameters
        fisher_dict, old_par_dict = get_fisher_dict(task_idx, train_loader, criterion, device)

    ### Mine: If seen classes is more than first 5 classes (0,1,2,3,4)
    else:
        print("Training style: EWC training")

        ### Mine: Train using EWC
        ewc_training(model, train_loader, optimizer, criterion, NUM_EPOCH, device, task_idx, fisher_dict, old_par_dict)

    # * ----------------------- eval ------------------------------
    print(f"\n -----------------------evaluation start-----------------------")
    accuracy = []
    for task_i, (_, _, x_test, t_test) in enumerate(dataset.task_data):
        if task_i > task_idx:
            continue

        current_task_classes = np.unique(t_test).tolist()
        x_test = torch.tensor(x_test).float()
        t_test = torch.tensor(t_test).long()
        test_data = TensorDataset(x_test, t_test)
        test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

        acc = evaluate_task(model, test_loader, current_task_classes)
        accuracy.append(acc)
    avg_accuracy = sum(accuracy) / len(accuracy)
    print(f"seen classes : {seen_classes_list} \t seen classes acc : {avg_accuracy:.6f}")
    print(f"-" * 50, "\n")

# torch.save(model.state_dict(), "Prob5.pth")
