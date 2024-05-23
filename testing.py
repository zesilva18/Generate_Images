import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import utils, datasets, transforms
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import torch.nn.functional as F

# Set random seed for reproducibility
torch.manual_seed(0)
device = torch.device('cuda:1' if (torch.cuda.is_available() and 1 > 0) else 'cpu')

# Batch size during training
BATCH_SIZE = 64

# Spatial size of training images. All images will be resized to this size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 1

# Number of classes in the training images. For mnist dataset this is 10
num_classes = 10

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 25

# Learning rate for optimizers
lr = 0.001

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

ngpu = 1

ndf = 64  # Size of feature maps in discriminator

nc = 3  # Number of channels in the training images (RGB)

# self, channel, out_channel, kernel_size, stride, padding

class Generator(nn.Module):
    def __init__(self, ngpu, nz, ngf, num_classes, nc):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.image = nn.Sequential(
            # state size. (nz) x 1 x 1
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True)
            # state size. (ngf*8) x 4 x 4
        )
        self.label = nn.Sequential(
            # state size. (num_classes) x 1 x 1
            nn.ConvTranspose2d(num_classes, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True)
            # state size. (ngf*8) x 4 x 4
        )
        self.main = nn.Sequential(
            # state size. (ngf*16) x 4 x 4
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 32 x 32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d(ngf, nc, 1, 1, 0, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, noise, labels):
        image = self.image(noise)
        label = self.label(labels)
        incat = torch.cat((image, label), dim=1)
        return self.main(incat)


class Discriminator(nn.Module):
    def __init__(self, ngpu, ndf, num_classes, nc):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.image = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
            # state size. (ndf) x 32 x 32
        )
        self.label = nn.Sequential(
            # input is (num_classes) x 64 x 64
            nn.Conv2d(num_classes, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
            # state size. (ndf) x 32 x 32
        )
        self.main = nn.Sequential(
            # state size. (ndf*2) x 32 x 32
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 16 x 16
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 8 x 8
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*16) x 4 x 4
            nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # state size. (1) x 1 x 1
        )

    def forward(self, image, label):
        image = self.image(image)
        label = self.label(label)
        incat = torch.cat((image, label), dim=1)
        return self.main(incat)


# Instantiate Generator and Discriminator
generator = Generator(ngpu, nz, ngf, num_classes, nc).to(device)
discriminator = Discriminator(ngpu, ndf, num_classes, nc).to(device)

# Create an example tensor for the Generator
noise = torch.randn(BATCH_SIZE, nz, 1, 1, device=device)
labels = torch.randint(0, num_classes, (BATCH_SIZE,), device=device)
labels_one_hot = F.one_hot(labels, num_classes).float().view(BATCH_SIZE, num_classes, 1, 1).to(device)

# Generate a fake image
fake_images = generator(noise, labels_one_hot)

# Test the Discriminator with the generated image
# Expand the labels to the size of the image
labels_expanded = F.one_hot(labels, num_classes).float().unsqueeze(2).unsqueeze(3).expand(BATCH_SIZE, num_classes, 64, 64).to(device)

print("Fake images shape:", fake_images.shape)

discriminator_output = discriminator(fake_images, labels_expanded)

# Output the results
print("Fake images shape:", fake_images.shape)  # Expected: [BATCH_SIZE, nc, 64, 64]
print("Discriminator output shape:", discriminator_output.shape)  # Expected: [BATCH_SIZE, 1, 1, 1]
