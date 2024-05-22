import re
import matplotlib.pyplot as plt
from PIL import Image # Install Pillow -> conda install anaconda::pillow or pip install pillow
import os
from skimage.io import  imread, imshow # Install scikit-image -> conda install scikit-image or pip install scikit-image
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
# import seaborn as sns
import pandas as pd
import torch.nn as nn
from torch.autograd import Variable
from torch import autograd
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from time import time


# General Settings
IMG_WIDTH = 75
IMG_HEIGHT = 75
BATCH_SIZE = 16

# Paths
train_dataset_path = 'data-students/TRAIN/'
test_dataset_path = 'data-students/TEST'

# Comments
"""
Instead of traing a seperate GAN model for each class, we can train a conditional GAN model where the generator and discriminator are conditioned on the class label. This is preferable because our dataset has very few samples for each class and training a seperate GAN model for each class would be computationally expensive and time consuming. As the image contain common features, like posts, background, etc, the knowledge learned by the GAN model for one class can be used to generate images for other classes as well. This is the main reason why we are using a conditional GAN model instead of training a seperate GAN model for each class.
"""


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, kernel_size, stride, padding):
            block = [
                nn.Conv2d(in_filters, out_filters, kernel_size=kernel_size, stride=stride, padding=padding), 
                nn.InstanceNorm2d(out_filters, 0.8),
                nn.LeakyReLU(0.2, inplace=True)
                ]

            return torch.nn.Sequential(*block)
        
        n = 128
        self.db_1 = discriminator_block(4, n, 3, 1, 0)
        self.db_2 = discriminator_block(n, 2*n, 5, 2, 1)
        self.db_3 = discriminator_block(2*n, 4*n, 4, 2, 1)
        self.db_4 = discriminator_block(4*n, 8*n, 4, 2, 0)
        self.adv_layer = nn.Sequential(nn.Conv2d(8*n, 1, 8, 1, 0))

        # Embedding for the label
        self.embedding = nn.Embedding(10, 24*24)
        self.transpose_embedding = nn.ConvTranspose2d(1, 1, 6, 3, 0)

    def forward(self, img, label):
        l = self.embedding(label)
        l = l.view(l.size(0), 1, 24, 24)
        l = self.transpose_embedding(l)
        x = torch.cat([img, l], 1)
        x = self.db_1(x)
        x = self.db_2(x)
        x = self.db_3(x)
        x = self.db_4(x)
        y = self.adv_layer(x)

        return y
    
    
class Generator(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.channels = 3
        n = 128

        def generator_block(in_filters, out_filters, kernel_size, stride, padding):
            block = [
                nn.ConvTranspose2d(in_filters, out_filters, kernel_size=kernel_size, stride=stride, padding=padding), 
                nn.BatchNorm2d(out_filters, 0.8),
                nn.ReLU(True)
                ]

            return torch.nn.Sequential(*block)
        
        self.gb_1 = generator_block(100+256, 16*n, 4, 1, 0)
        self.gb_2 = generator_block(16*n, 8*n, 4, 2, 1)
        self.gb_3 = generator_block(8*n, 4*n, 4, 2, 0)
        self.gb_4 = generator_block(4*n, 2*n, 4, 2, 1)
        self.gb_5 = generator_block(2*n, n, 5, 2, 1)
        self.projection_layer = torch.nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=n, 
                out_channels=self.channels, 
                kernel_size=3, 
                stride=1, 
                padding=0
                ),
            nn.Tanh()
        )

        self.embedding = nn.Embedding(10, 256)

    def forward(self, noise, label):
        l = self.embedding(label)
        l = l.view(l.size(0), l.size(1), 1, 1)
        x = torch.cat([noise, l], 1)
        x = self.gb_1(x)
        x = self.gb_2(x)
        x = self.gb_3(x)
        x = self.gb_4(x)
        x = self.gb_5(x)
        y = self.projection_layer(x)
        return y
    
    def make_random_noise_vector(self, batch_size):
        return torch.randn(batch_size, 100, 1, 1)


class WGAN(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        """
        WGAN or Wasserstein GAN model is a type of GAN model that uses the Wasserstein distance to train the generator and discriminator. It improves the stability of the GAN model and helps in generating better quality images. In this architecture the discriminator has a linear output activation, that represents the quality of the generated images instead of a simple classification. Has the quality tends to improve gradient penalty is used to enforce the Lipschitz constraint on the discriminator. This helps in generating better quality images and improves the stability of the GAN model.

        A conditional Deep Convolutional GAN (cDCGAN) architecture is used for the generator and discriminator. The generator uses a series of transpose convolutional layers to generate images from a random noise vector and a label. The discriminator uses a series of convolutional layers to classify the images as real or fake. The discriminator also uses an embedding layer to embed the class label into the image. This helps in generating images for a specific class.
        """
        super().__init__(*args, **kwargs)
        
        self.generator = Generator()
        self.discriminator = Discriminator()

        self.generator = self.generator.to('cuda')
        self.discriminator = self.discriminator.to('cuda')
    
    def forward(self, label):
        noise = self.generator.make_random_noise_vector(1).to('cuda')
        fake_image = self.generator(noise, label)
        fake_image = fake_image.cpu().detach().numpy()
        fake_image = (fake_image + 1) / 2
        return fake_image.transpose(0, 2, 3, 1)[0]

    def gradient_penalty(self, real_images, fake_images, label):
        batch_size = real_images.size(0)
        eta = torch.FloatTensor(batch_size,1,1,1).uniform_(0,1)
        eta = eta.expand(batch_size, real_images.size(1), real_images.size(2), real_images.size(3)).to('cuda')

        interpolated = eta * real_images + ((1 - eta) * fake_images)
        interpolated = interpolated.to('cuda')

        # define it to calculate gradient
        interpolated = Variable(interpolated, requires_grad=True).to('cuda')

        # calculate probability of interpolated examples
        prob_interpolated = self.discriminator(interpolated, label).to('cuda')

        # calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(
            outputs=prob_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones(prob_interpolated.size()).to('cuda'),
            create_graph=True,
            retain_graph=True)[0].to('cuda')

        # flatten the gradients to it calculates norm batchwise
        gradients = gradients.view(gradients.size(0), -1)
        
        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return grad_penalty
    
    def train(self, dataset, n_epochs: int, batch_size: int, n_critic: int, lr_generator: float, lr_discriminator: float):
        """
        Train the WGAN
        """
        # Define the optimizer
        optimizer_discriminator = torch.optim.AdamW(self.discriminator.parameters(), lr=lr_discriminator, betas=(0.5, 0.999))
        optimizer_generator = torch.optim.AdamW(self.generator.parameters(), lr=lr_generator, betas=(0.5, 0.999))

        for epoch in range(n_epochs):
            pbar = tqdm(enumerate(dataset), total=len(dataset))
            for i, (imgs, label) in enumerate(dataset):
                label = label.to('cuda')
                imgs = imgs.to('cuda')
    
                batch_size = imgs.shape[0]

                for p in self.discriminator.parameters():
                        p.requires_grad = True

                # Train the discriminator
                for _ in range(n_critic):
                    self.discriminator.zero_grad()

                    # Generate a batch of images
                    noise = self.generator.make_random_noise_vector(batch_size).to('cuda')

                    # Loss for real images
                    d_real_loss = self.discriminator(imgs, label)
                    d_real_loss = d_real_loss.mean()

                    # Loss for fake images
                    fake_images = self.generator(noise, label).detach()
                    d_fake_loss = self.discriminator(fake_images, label)
                    d_fake_loss = d_fake_loss.mean()

                    w_d = d_real_loss - d_fake_loss
                    # Gradient penalty
                    gradient_penalty = self.gradient_penalty(imgs.data, fake_images.data, label)

                    # Total loss
                    d_loss = -w_d + gradient_penalty * 5
                    d_loss.backward()
                    optimizer_discriminator.step()
                
                # Train the generator
                for p in self.discriminator.parameters():
                    p.requires_grad = False

                self.generator.zero_grad()

                noise = self.generator.make_random_noise_vector(batch_size).to('cuda')

                fake_images = self.generator(noise, label)
                g_loss = self.discriminator(fake_images, label)
                g_loss = -g_loss.mean()
                g_loss.backward()
                g_cost = -g_loss
                optimizer_generator.step()

                pbar.set_description(f'Epoch: {epoch}, Batch: {i}, Discriminator Loss: {d_loss}, Generator Loss: {g_cost}')

                l = torch.tensor([0]).to('cuda')
                image = self.forward(l)
                plt.imshow(image)
                plt.savefig(f'output/output.png')

            if epoch % 10 == 0:
                noise = self.generator.make_random_noise_vector(25).to('cuda')
                label = torch.randint(0, 10, (25,)).to('cuda')
                print(label)
                fake_images = self.generator(noise, label)
                fake_images = fake_images.cpu().detach().numpy()
                fig, axes = plt.subplots(5, 5, figsize=(10, 10))
                for i, ax in enumerate(axes.flat):
                    fake_images[i] = (fake_images[i] + 1) / 2
                    ax.imshow(fake_images[i].transpose(1, 2, 0))
                    ax.axis('off')
                    ax.set_title(f'Label: {label[i]}')
                plt.savefig(f'output/epoch_{epoch}.png')
                self.save(f'output/epoch_{epoch}', save_generator=True, save_discriminator=True)

    def save(self, path, save_generator=True, save_discriminator=True):
        if save_generator:
            torch.save(self.generator.state_dict(), path + '_generator.pth')
        if save_discriminator:
            torch.save(self.discriminator.state_dict(), path + '_discriminator.pth')

    def load(self, path, load_generator=True, load_discriminator=True):
        if load_generator:
            self.generator.load_state_dict(torch.load(path + '_generator.pth'))
        if load_discriminator:
            self.discriminator.load_state_dict(torch.load(path + '_discriminator.pth'))


if __name__ == '__main__':
    wgan = WGAN()

    transform = transforms.Compose(
        [transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),transforms.ToTensor()])
    traffic_signals_dataset = datasets.ImageFolder(root=train_dataset_path, transform=transform)
    
    train_idx, valid_idx = train_test_split(
        range(len(traffic_signals_dataset)),
        test_size=0.1,
        shuffle=True,
        stratify=traffic_signals_dataset.targets
    )

    train_subset = Subset(traffic_signals_dataset, train_idx)
    valid_subset = Subset(traffic_signals_dataset, valid_idx)

    train_dataset_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    validation_dataset_loader = DataLoader(valid_subset, batch_size=BATCH_SIZE, shuffle=False)

    training_targets = traffic_signals_dataset.targets
    t_targets = {k:0 for k in training_targets}
    for t in training_targets:
        t_targets[t] += 1
    print('Training class distribution:', t_targets)

    wgan.train(train_dataset_loader, n_epochs=200, batch_size=BATCH_SIZE, n_critic=5, lr_generator=1e-4, lr_discriminator=1e-4)
