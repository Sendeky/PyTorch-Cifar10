import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

from cifar10 import Cifar10
from CustomCIFAR10Dataset import CustomCIFAR10Dataset


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
ngpu = 1

# number of workers
nw = 0

# number of training epochs
num_epochs = 33 

# learning rate
learning_rate = 0.0022

# chooses which device to use
device = torch.device("cuda:0" if (torch.cuda.is_available()) and (ngpu > 0) else "cpu")

# transforms for image. CONVERT TO TENSOR VERY IMPORTANT, OTHERWISE DATALOADER WON"T ACCEPT IMAGE
transform = transforms.Compose([
    transforms.Resize((32, 32)),                # Resize the image to 32x32 (required for CIFAR-10)
    transforms.RandomResizedCrop(32, scale=(0.8, 0.8)),    # Random crop (image augmentation)
    transforms.RandomHorizontalFlip(),          # Random hoizontal flip (image augmentation) 
    transforms.ToTensor(),                      # Convert PIL Image to a tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize the image to [-1, 1]
])

# We use our custsom cifar10 dataset class to convert the dataset to a format that the torch dataloader can use
train_ds = CustomCIFAR10Dataset(train_data["img"], train_data["label"], transform=transform)
test_ds = CustomCIFAR10Dataset(test_data["img"], test_data["label"], transform=transform)

# LOADERS FOR DATASET
train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=nw)
test_loader = DataLoader(test_ds, batch_size, shuffle=True, num_workers=nw)


# The nueral net class
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.network = nn.Sequential(
            # first 2 concolutional layers
            nn.Conv2d(nc, 16, kernel_size=3, stride=1, padding=1),          # a convoltional layer with 3 input channels, 16 output channels,
                                                                            # a kernel size of 3, a stride of 1, and padding of 1
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=1, stride=1),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.MaxPool2d(1, 1),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=1, stride=1),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=1, stride=1),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),

            # max pooling layers
            nn.MaxPool2d(kernel_size=2, stride=2),                          # a max pooling layer with kernel size of 3 and stride of 2
                                                                            # helps reduce spatial dimensions of feature maps
            nn.Flatten(),
            nn.Linear(16 * 16 * 16, 64),                                    # adjust the input size based on the output of the last conv layer
            nn.Linear(64, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 16),
            nn.Linear(16, output),
        )


    def forward(self, x):
        return self.network(x)
    
# creates instance of the model
model = Net()

print("model parameters: ", model.parameters)

# create the optimizer and criterion
criterion = nn.CrossEntropyLoss()
# Adam optimizer yields much better results than SGD
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)

# moves model to device (ie. cpu/gpu)
model.to(device)

# accuracy array (for plotting)
accuracyArr = []

print("started training")
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
        
    print(f"epoch: {epoch + 1}/{num_epochs} Loss: {running_loss}")

    # After training, evaluate the model on the test dataset to get final performance metrics
    model.eval()  # Set the model to evaluation mode (important when using dropout or batch normalization)
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            predictions = model(images)

            # Compute evaluation metrics (e.g., accuracy, precision, recall, etc.)
            # get predicted class for each image
            _, predicted = torch.max(predictions.data, 1)

            # Count the total number of labels in the test dataset
            total += labels.size(0)

            # Count the number of correct predictions
            correct += (predicted == labels).sum().item()

    # calculate the accuracy
    accuracy = correct/total
    accuracyArr.append(accuracy)
    print(f"Accuracy on the test dataset: {accuracy:.2%}, epoch: {epoch + 1}")


# After training, evaluate the model on the test dataset to get final performance metrics
model.eval()  # Set the model to evaluation mode (important when using dropout or batch normalization)
correct = 0
total = 0

with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        predictions = model(images)

        # Compute evaluation metrics (e.g., accuracy, precision, recall, etc.)
        # get predicted class for each image
        _, predicted = torch.max(predictions.data, 1)

        # Count the total number of labels in the test dataset
        total += labels.size(0)

        # Count the number of correct predictions
        correct += (predicted == labels).sum().item()

# calculate the accuracy
accuracy = correct/total
print(f"Accuracy on the test dataset: {accuracy:.2%}")


## IMPROVEMENTS/DEGREDATIONS ##
# BASELINE: ~51-54%

# After AutoAugment(CIFAR10):  ~40%

# After Dropout: ~51-52%

# After adding another fully connected layer (64 in, 16 out): ~50-51%

# After adding weight decay to optimizer: (0.01): ~51+%

# ADDED: After adding all layers to nn.Sequential: ~55-57%

# ADDED: After using optim.Adam instead of optim.SGD: ~61-62%

# ADDED: After mimicking VGG16 architecture with: Conv2d -> ReLU -> MaxPool -> REPEAT

# MAX CURRENT ACCURACY: 72.71%

# import numpy as np
# import matplotlib.pyplot as plt
# y = np.array(accuracyArr)
# x = num_epochs
# plt.plot(x, y)
# plt.show()