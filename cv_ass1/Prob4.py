# Import torch for testing model
import torch

# Import Dataset class to pair image and labels and Dataloader to load data in batches for training
from torch.utils.data import DataLoader

# Import model
from Prob2 import MyModel

# Import PIL to load images
from PIL import Image

# Import torch transformations to manipulate the images
import torchvision.transforms as T

# Define path to the model
model_path = "Prob3.pth"

# Define the device to use
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define the transformation for the image
my_transform = T.Compose(
    [  # Convert the image array to tensor
        T.ToTensor(),
        # Center crop the image
        T.CenterCrop((64, 64)),
    ]
)


# Function for testing
def prob4(img_path):
    # Load image
    image = Image.open(img_path)

    # Transform image
    my_image = my_transform(image)

    # Define model
    model = MyModel(3, 12, 60)

    # Load the weights and biases of the model
    model.load_state_dict(torch.load(model_path))

    # Load model to preferred device
    model = model.to(device)

    # Set model to test
    model.eval()

    # Change the shape of the image to have one dimesion of batch ([1, 3, 64, 64])
    my_image = my_image.unsqueeze(0)

    # When necessary print dimension of image
    # print(my_image.shape)

    # Set torch to no grad for faster computation
    with torch.no_grad():
        # Predict the hour and minute for the input image which is also sent to device
        val_pred_h, val_pred_m = model(my_image.to(device))
        # Obtain the highest probability hour
        _, predicted_h = torch.max(val_pred_h.data, 1)
        # Obtain the highest probability minute
        _, predicted_m = torch.max(val_pred_m.data, 1)
        # Obtain hour from the tensor
        hour = predicted_h.item()
        # Obtain minute from the tensor
        minute = predicted_m.item()

    #### OUTPUT HOUR MINUTE

    return hour, minute


########## EXAMPLE
# img_path = "/home/xiyana/Documents/clock/version_1/1h 16m.jpg"
# hour, minute = prob4(img_path)
# print(hour, minute)
