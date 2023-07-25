from PIL import Image
from torch.utils.data import Dataset

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
