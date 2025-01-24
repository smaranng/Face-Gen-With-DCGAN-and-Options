import streamlit as st
import zipfile
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# File extraction section
st.title("DCGAN Image Generator")
uploaded_file = st.file_uploader("Upload CelebA Dataset (ZIP format)", type="zip")

# Parameters for model
image_size = 64
batch_size = 32
noise_dim = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize dataloader
dataloader = None

if uploaded_file:
    with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
        zip_ref.extractall("dataset")
    st.success("Dataset extracted successfully!")

    # Dataset loading section
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset_path = "dataset/img_align_celeba"
    if os.path.exists(dataset_path):
        dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        st.write("Dataset loaded successfully!")

# Discriminator Model
class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid()
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)

# Generator Model
class Generator(nn.Module):
    def __init__(self, noise_dim, channels_img, features_g):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            self._block(noise_dim, features_g * 16, 4, 1, 0),
            self._block(features_g * 16, features_g * 8, 4, 2, 1),
            self._block(features_g * 8, features_g * 4, 4, 2, 1),
            self._block(features_g * 4, features_g * 2, 4, 2, 1),
            nn.ConvTranspose2d(features_g * 2, channels_img, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.gen(x)

# Instantiate models
gen = Generator(noise_dim=noise_dim, channels_img=3, features_g=64).to(device)
disc = Discriminator(channels_img=3, features_d=64).to(device)

# Optimizers and loss function
opt_gen = optim.Adam(gen.parameters(), lr=0.0002, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=0.0002, betas=(0.5, 0.999))
criterion = nn.BCELoss()

# Training function
def train_dcgan(num_epochs):
    if not dataloader:
        st.error("No dataset available for training. Please upload and extract the dataset.")
        return
    
    st.write("Training started...")
    fixed_noise = torch.randn(32, noise_dim, 1, 1).to(device)
    
    for epoch in range(num_epochs):
        for batch_idx, (real, _) in enumerate(dataloader):
            real = real.to(device)
            noise = torch.randn(batch_size, noise_dim, 1, 1).to(device)
            fake = gen(noise)

            # Train Discriminator
            disc_real = disc(real).reshape(-1)
            loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = disc(fake.detach()).reshape(-1)
            loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            loss_disc = (loss_disc_real + loss_disc_fake) / 2
            disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()

            # Train Generator
            output = disc(fake).reshape(-1)
            loss_gen = criterion(output, torch.ones_like(output))
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

        st.write(f"Epoch [{epoch+1}/{num_epochs}] Loss D: {loss_disc:.4f}, Loss G: {loss_gen:.4f}")

        if epoch % 5 == 0:
            with torch.no_grad():
                fake = gen(fixed_noise).detach().cpu()
                grid = torchvision.utils.make_grid(fake, nrow=8, normalize=True)
                np_grid = grid.numpy().transpose((1, 2, 0))
                plt.imshow(np_grid)
                st.pyplot(plt)

# Control the number of epochs
num_epochs = st.slider('Select number of epochs', 1, 100, 10)

# Start training button
if st.button("Start Training"):
    train_dcgan(num_epochs)
