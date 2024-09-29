# Load torch nn to create model (i.e, nn.Module, conv)
import torch.nn as nn


# Define model class
class MyModel(nn.Module):
    # Define init function and the required parameters
    def __init__(self, in_channels, num_classes_hour, num_classes_minutes):
        # Init model class
        super().__init__()
        # Define first convolution layer with 16 output channels
        self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=2, stride=1, padding=1)
        # Define first convolution layer with 32 output channels
        self.conv_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, stride=1, padding=1)

        # Define pooling
        self.avg_pl = nn.AvgPool2d(kernel_size=2, stride=2)

        # Define batch normalization for each conv layer
        self.btc_n_1 = nn.BatchNorm2d(num_features=16)
        self.btc_n_2 = nn.BatchNorm2d(num_features=32)

        # Define the activation function
        self.acti_fnc = nn.ReLU()

        # Define the first linear layer
        self.fc_1 = nn.Linear(in_features=8192, out_features=512)

        # Define hour classifier
        self.classifier_h = nn.Linear(in_features=512, out_features=num_classes_hour)

        # Define minute classifier
        self.classifier_m = nn.Linear(in_features=512, out_features=num_classes_minutes)

    # Define forward function
    def forward(self, img):
        # Perform first block of convolution-> bacth norm -> activation function -> pooling
        out = self.avg_pl(self.acti_fnc(self.btc_n_1(self.conv_1(img))))
        # Perform second block of convolution-> bacth norm -> activation function -> pooling
        out = self.avg_pl(self.acti_fnc(self.btc_n_2(self.conv_2(out))))

        # Reshape the output to fit it to the fully connected layer
        out = out.view(out.size(0), -1)

        # Pass output to first fc
        out = self.fc_1(out)

        # Pass output to activation function
        out = self.acti_fnc(out)

        # Pass output to hour classifier
        out_h = self.classifier_h(out)

        # Pass output to minute classifier
        out_m = self.classifier_m(out)

        return out_h, out_m
