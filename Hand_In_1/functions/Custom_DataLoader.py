

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import Dataset


class InsectsDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.insects_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.insects_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.insects_frame.iloc[idx, 2])  # Assuming filename is in the second column
        image = read_image(img_name)  # Using torchvision.io to read the image
        species = self.insects_frame.iloc[idx, 1]  # Assuming species is in the first column

        # Transform picture to correct size
        if self.transform:
            image = self.transform(image)

        return [image, species]
    


class ConditionalToTensor(transforms.ToTensor):
    def __call__(self, pic):
        if isinstance(pic, torch.Tensor):
            return pic
        return super().__call__(pic)



def custom_dataLoader():
    # Detect if a GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define a basic transform to convert the images to PyTorch tensors
    transform = transforms.Compose([
        transforms.Resize((128, 128)), # Resize images for demonstration purposes
        ConditionalToTensor() # Checks if the input is a tensor before converting it
    ])

    csv_file='./data/Insects.csv'
    Image_folder='./data/Insects/'

    # Create an instance of the dataset
    insects_dataset = InsectsDataset(csv_file=csv_file, root_dir=Image_folder, transform=transform)

    batches = 4

    # Set up the dataset.
    trainloader = torch.utils.data.DataLoader(insects_dataset, 
                                            batch_size=batches, 
                                            shuffle=True, 
                                            num_workers=2)

    # get some images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # Move data (tensors) to the GPU
    images = images.to(device)

    # How many batches to show
    num_examples = 5

    # Scale the figure size based on the number of batches and examples
    scale_of_figure = 3

    # Adjust the number of columns and rows to display, together with the figsize,
    # given how large the batches are and the number of examples we want to show
    fig, axs = plt.subplots(num_examples, batches, 
                            figsize=(scale_of_figure * batches, scale_of_figure * num_examples)) 

    img_count = 0
    for i in range(num_examples): # Run through 5 batches
        images, labels = next(dataiter)
        images = images.to(device) # Move the tensors to the GPU
        for image, label in zip(images, labels): # Run through all samples in a batch
            row = img_count // batches
            col = img_count % batches
            axs[row, col].imshow(np.transpose(image.cpu().numpy(), (1, 2, 0)))
            axs[row, col].set_title(label)
            axs[row, col].axis('off')
            img_count += 1

    # If there are unused subplot axes at the end, turn them off
    for idx in range(img_count, num_examples * batches):
        row = idx // batches
        col = idx % batches
        axs[row, col].axis('off')

    plt.tight_layout()
    plt.show()
