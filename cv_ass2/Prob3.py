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

### Mine: Define two variables to accumulate the training data
accumulated_data_x = None
accumulated_data_t = None

# * ----------------------- task process ------------------------------
seen_classes_list = []  # Renamed from accumulate
for task_idx, (x_train_task, t_train_task, x_test_task, t_test_task) in enumerate(dataset.task_data):
    # Update seen_classes_list
    seen_classes = np.unique(t_train_task).tolist()
    seen_classes_list.extend(seen_classes)
    seen_classes_list = list(
        set(seen_classes_list)
    )  # if use the replay method, for preventing ovelapping each classes

    ### Mine: If seen classes is only first 5 classes (0,1,2,3,4)
    if len(seen_classes_list) == 5:
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

    ### Mine: If seen classes is only first 5 classes (0,1,2,3,4)
    if task_idx == 0:
        ### Mine: accumulate the sample data
        accumulated_data_x = x_train_task

        ### Mine: accumulate the target data
        accumulated_data_t = t_train_task

    ### Mine: If seen classes is more than first 5 classes (0,1,2,3,4)
    else:
        ### Mine: concatenate the accumulated data
        print(accumulated_data_t.size())
        print(t_train_task.size())
        accumulated_data_x = torch.concat((accumulated_data_x, x_train_task), axis=0)
        accumulated_data_t = torch.concat((accumulated_data_t, t_train_task), axis=0)

    train_dataset = TensorDataset(accumulated_data_x, accumulated_data_t)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # * ----------------------- train ------------------------------

    train_task(model, train_loader, optimizer, criterion, NUM_EPOCH)

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


# torch.save(model.state_dict(), "Prob3.pth")
