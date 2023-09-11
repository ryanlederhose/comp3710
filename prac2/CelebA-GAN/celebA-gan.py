import torch
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import os
import torch.nn.functional as F
from tqdm import tqdm

# Directory
DIR = 'C:/Users/ryanl/comp3710/prac2/CelebA-GAN/input/'
SAMPLE_DIR = 'C:/Users/ryanl/comp3710/prac2/CelebA-GAN/output/'
os.makedirs(SAMPLE_DIR, exist_ok=True)

# Parameters
batch_size = 128
image_size = 64
normalisation = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
latent_size = 128
lr = 0.00025
epochs = 60

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Define the discriminator
class Discriminator(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.batch1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.batch2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.batch3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.batch4 = nn.BatchNorm2d(512)

        self.conv5 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False)

        self.leakyrelu = nn.LeakyReLU(0.2, inplace=True)

        self.flatten = nn.Flatten()
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.leakyrelu(self.batch1(self.conv1(x)))
        x = self.leakyrelu(self.batch2(self.conv2(x)))
        x = self.leakyrelu(self.batch3(self.conv3(x)))
        x = self.leakyrelu(self.batch4(self.conv4(x)))
        x = self.conv5(x)
        x = self.sigmoid(self.flatten(x))
        return x
    
discriminator = Discriminator()
discriminator.to(device)

# Define the generator
class Generator(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.conv1 = nn.ConvTranspose2d(latent_size, 512, kernel_size=4, stride=1, padding=0, bias=False)
        self.batch1 = nn.BatchNorm2d(512)

        self.conv2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.batch2 = nn.BatchNorm2d(256)

        self.conv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.batch3 = nn.BatchNorm2d(128)

        self.conv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.batch4 = nn.BatchNorm2d(64)

        self.conv5 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False)
        
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.relu(self.batch1(self.conv1(x)))
        x = self.relu(self.batch2(self.conv2(x)))
        x = self.relu(self.batch3(self.conv3(x)))
        x = self.relu(self.batch4(self.conv4(x)))
        x = self.tanh(self.conv5(x))
        return x
    
generator = Generator()
generator.to(device)
        

'''
Denormalise the image tensor
'''
def denorm(img_tensors):
    return img_tensors * normalisation[1][0] + normalisation[0][0]

'''
Save generated image samples
'''
def save_samples(index, latent_tensors):
    image = generator(latent_tensors)
    file = '{0:0=4d}.png'.format(index)
    save_image(denorm(image), os.path.join(SAMPLE_DIR, file), nrow=8)

'''
Train the discrimnator against real images
'''
def train_discriminator(real_images, discriminator_optimizer):
    discriminator_optimizer.zero_grad()

    # Pass real images through  discriminator
    real_preds = discriminator(real_images)
    real_targets = torch.ones(real_images.size(0), 1, device=device)
    real_loss = F.binary_cross_entropy(real_preds, real_targets)
    real_score = torch.mean(real_preds).item()

    # Generate images
    generated_images = generator(torch.randn(batch_size, latent_size, 1, 1, device=device))

    # Pass generated images through discriminator
    fake_targets = torch.zeros(generated_images.size(0), 1, device=device)
    fake_preds = discriminator(generated_images)
    fake_loss = F.binary_cross_entropy(fake_preds, fake_targets)
    fake_score = torch.mean(fake_preds).item()

    # Update discriminator weights
    loss = real_loss + fake_loss
    loss.backward()
    discriminator_optimizer.step()
    return loss.item(), real_score, fake_score

'''
Train the generator
'''
def train_generator(generator_optimizer):
    generator_optimizer.zero_grad()

    # Generate images
    generated_images = generator(torch.randn(batch_size, latent_size, 1, 1, device=device))

    # Use the discriminator 
    predictions = discriminator(generated_images)
    targets = torch.ones(batch_size, 1, device=device)
    loss = F.binary_cross_entropy(predictions, targets)

    # Update generator 
    loss.backward()
    generator_optimizer.step()

    return loss.item()

'''
Fit the GAN to the training data and generate images
'''
def train(epochs, lr, trainloader, fixed_latent, start_idx = 1):

    generatorLosses = []
    discriminatorLosses = []
    real_scores = []
    fake_scores = []

    # Define optimisers
    discriminator_optimizer = Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    generator_optimizer = Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))

    # Train models
    for epoch in range(epochs):
        for batch_idx, (data, targets) in enumerate(tqdm(trainloader)):
            dLoss, real_score, fake_score = train_discriminator(data.to(device), discriminator_optimizer)
            gLoss = train_generator(generator_optimizer)

        generatorLosses.append(gLoss)
        discriminatorLosses.append(dLoss)
        real_scores.append(real_score)
        fake_scores.append(fake_score)

        print("Epoch [{}/{}]".format(epoch+1, epochs))
        save_samples(epoch+start_idx, fixed_latent)

    return generatorLosses, discriminatorLosses, real_scores, fake_scores

def main():

    # Get training dataset from image folder
    tran_images = ImageFolder(root=DIR, transform=T.Compose([T.Resize(image_size), 
                                                             T.CenterCrop(image_size), T.ToTensor(), T.Normalize(*normalisation)]))
    
    # Get training loader
    trainloader = DataLoader(tran_images, batch_size=batch_size, shuffle=True, num_workers=3, pin_memory=True)

    fixed_latent = torch.randn(64, latent_size, 1, 1, device=device)
    save_samples(0, fixed_latent)

    # Train generator and discriminator
    generatorLosses, discriminatorLosses, real_scores, fake_scores = train(epochs, lr, trainloader, fixed_latent)

    # Save models
    torch.save(generator.state_dict(), 'G.pth')
    torch.save(discriminator.state_dict(), 'D.pth')

    # Plot discriminator and generator losses
    plt.plot(discriminatorLosses, '-')
    plt.plot(generatorLosses, '-')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Discriminator', 'Generator'])
    plt.title('Losses')
    plt.show()

    # Plot scores of real and generated images
    plt.plot(real_scores, '-')
    plt.plot(fake_scores, '-')
    plt.xlabel('epoch')
    plt.ylabel('score')
    plt.legend(['Real', 'Fake'])
    plt.title('Scores')
    plt.show()


if __name__ == '__main__':
    main()