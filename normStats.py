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

augs = A.Compose([A.Resize(height = image_size, 
                           width  = image_size),
                  A.Normalize(mean = (0, 0, 0),
                              std  = (1, 1, 1)),
                  ToTensorV2()])

# We use our custsom cifar10 dataset class to convert the dataset to a format that the torch dataloader can use
train_ds = CustomCIFAR10Dataset(train_data["img"], train_data["label"], transform=transform)
test_ds = CustomCIFAR10Dataset(test_data["img"], test_data["label"], transform=transform)

# LOADERS FOR DATASET
train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=nw)
test_loader = DataLoader(test_ds, batch_size, shuffle=True, num_workers=nw)

# placeholders
psum    = torch.tensor([0.0, 0.0, 0.0])
psum_sq = torch.tensor([0.0, 0.0, 0.0])

# loop through images
for inputs in tqdm(train_loader):
    psum    += inputs.sum(axis        = [0, 2, 3])
    psum_sq += (inputs ** 2).sum(axis = [0, 2, 3])



####### FINAL CALCULATIONS

# pixel count
count = len(df) * image_size * image_size

# mean and std
total_mean = psum / count
total_var  = (psum_sq / count) - (total_mean ** 2)
total_std  = torch.sqrt(total_var)

# output
print('mean: '  + str(total_mean))
print('std:  '  + str(total_std))