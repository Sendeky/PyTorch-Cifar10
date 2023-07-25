import os
import datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from cifar10 import Cifar10


# create dataset builder instance
cifar10_builder = Cifar10()
# downloads the dataset
cifar10_builder.download_and_prepare()

# generate the dataset ('train', 'test' portion)
train_data = cifar10_builder.as_dataset(split='train')
test_data = cifar10_builder.as_dataset(split='test')

train_images = train_data["img"]
train_labels = train_data["label"]

test_images = test_data["img"]
test_labels = test_data["label"]

# Cifar10 classes
classes =  ("airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck")

# # we can plot and access the images like this
# from matplotlib import pyplot as plt
# from matplotlib import image as mpimg

# # doing index first and then "img" is faster because image is decoded immediately when chosen (index -> decoding is faster than decoding -> index)
# plt.imshow(train_ds[0]["img"])
# plt.show()

# We have to make a custom dataset class to load them with the torch DataLoader
# Custom dataset class for CIFAR-10 images
class CustomCIFAR10Dataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        # Apply the transformations (if any)
        if self.transform is not None:
            image = self.transform(image)

        return image, label


# PARAMETERS

# batch size during training
batch_size = 128

# image size
img_size = 32

# number of channels in image (3, because RGB in this case)
nc = 3

# output size (10 classes)
output = len(classes)

# Num of GPUs (pick 0 for CPU)
ngpu = 0

# number of workers
nw = 0

# number of training epochs
num_epochs = 5

# learning rate
learning_rate = 0.0022

# chooses which device to use
device = torch.device("cuda:0" if (torch.cuda.is_available()) and (ngpu > 0) else "cpu")

# transforms for image. CONVERT TO TENSOR VERY IMPORTANT, OTHERWISE DATALOADER WON"T ACCEPT IMAGE
transform = transforms.Compose([
    transforms.Resize((32, 32)),         # Resize the image to 32x32 (required for CIFAR-10)
    transforms.ToTensor(),               # Convert PIL Image to a tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize the image to [-1, 1]
])

train_ds = CustomCIFAR10Dataset(train_data["img"], train_data["label"], transform=transform)
test_ds = CustomCIFAR10Dataset(test_data["img"], test_data["label"], transform=transform)

# LOADERS FOR DATASET
train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=nw)
test_loader = DataLoader(test_ds, batch_size, shuffle=True, num_workers=nw)


# The nueral net class
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # first 2 concolutional layers
        self.conv1 = nn.Conv2d(nc, 16, kernel_size=3, stride=1, padding=1)         # a convoltional layer with 3 input channels, 16 output channels,
                                                            # a kernel size of 3, a stride of 1, and padding of 1
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)

        # max pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)       # a max pooling layer with kernel size of 3 and stride of 1
                                                            # helps reduce spatial dimensions of feature maps

        self.fc1 = nn.Linear(32 * 8 * 8, 16)                   # adjust the input size based on the output of the last conv layer
        self.fc2 = nn.Linear(16, output)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))                # First convoltional layer, then ReLU active, then max pooling
        x = self.pool(F.relu(self.conv2(x)))                # Second convolutional layer, then ReLu, then pooling

        x = x.view(x.size(0), -1)                           # Flatten tensor before passing through fully connected layers

        x = F.relu(self.fc1(x))                          # First fully connected layer, then ReLu, then pooling
        x = self.fc2(x)                                     # Layer with predictions, fully connected

        return x
    
# creates instance of the model
model = Net()

# create the optimizer and criterion
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters() , lr=learning_rate, momentum=0.9)

# moves model to device (ie. cpu/gpu)
model.to(device)


for epoch in range(num_epochs):
    model.train()           # set model to training mode (important when using dropout or batch normalization)

    running_loss = 0.0
    for batch_idx, (images, labels) in enumerate(train_loader):
        inputs = images.to(device)
        labels = labels.to(device)
        # print("print inputs shape: ", inputs.shape)

        optimizer.zero_grad()       # reset gradients

        # forward pass
        predictions = model(inputs)

        # compute loss
        loss = criterion(predictions, labels)

        # Backpropogation
        loss.backward()

        # update models parameters
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if batch_idx % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {batch_idx + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
        
print("finished training")


# # After training, evaluate the model on the test dataset to get final performance metrics
# model.eval()  # Set the model to evaluation mode (important when using dropout or batch normalization)
# with torch.no_grad():
#     for batch_idx, (images, labels) in enumerate(test_ds):
#         images = images.to(device)
#         labels = labels.to(device)

#         # Forward pass
#         predictions = model(images)

#         # Compute evaluation metrics (e.g., accuracy, precision, recall, etc.)