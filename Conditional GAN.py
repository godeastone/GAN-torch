import os
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image, ImageFont, ImageDraw


# Hyper-parameters & Variables setting
num_epoch = 200
batch_size = 100
learning_rate = 0.0002
img_size = 28 * 28
num_channel = 1
dir_name = "CGAN_results"

noise_size = 100
hidden_size1 = 256
hidden_size2 = 512

"""
FOR CONDITIONAL GAN
"""
# The number of MNIST's class label is 10
condition_size = 10


# Device setting
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Now using {} devices".format(device))


# Create a directory for saving samples
if not os.path.exists(dir_name):
    os.makedirs(dir_name)


# Dataset transform setting
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)])

# MNIST dataset setting
MNIST_dataset = torchvision.datasets.MNIST(root='../../data/',
                                           train=True,
                                           transform=transform,
                                           download=True)

# Data loader
data_loader = torch.utils.data.DataLoader(dataset=MNIST_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)


# Declares discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.linear1 = nn.Linear(img_size + condition_size, hidden_size2)
        self.linear2 = nn.Linear(hidden_size2, hidden_size1)
        self.linear3 = nn.Linear(hidden_size1, 1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.leaky_relu(self.linear1(x))
        x = self.leaky_relu(self.linear2(x))
        x = self.linear3(x)
        x = self.sigmoid(x)
        return x


# Declares generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.linear1 = nn.Linear(noise_size + condition_size, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, img_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        x = self.tanh(x)
        return x


# Initialize generator/Discriminator
discriminator = Discriminator()
generator = Generator()

# Device setting
discriminator = discriminator.to(device)
generator = generator.to(device)

# Loss function & Optimizer setting
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)


"""
Training part
"""
for epoch in range(num_epoch):
    for i, (images, label) in enumerate(data_loader):

        # make ground truth (labels) -> 1 for real, 0 for fake
        real_label = torch.full((batch_size, 1), 1, dtype=torch.float32).to(device)
        fake_label = torch.full((batch_size, 1), 0, dtype=torch.float32).to(device)

        # reshape real images from MNIST dataset
        real_images = images.reshape(batch_size, -1).to(device)

        """
        FOR CONDITIONAL GAN
        """
        # Encode MNIST's label's with 'one hot encoding'
        label_encoded = F.one_hot(label, num_classes=10).to(device)
        # concat real images with 'label encoded vector'
        real_images_concat = torch.cat((real_images, label_encoded), 1)

        # +---------------------+
        # |   train Generator   |
        # +---------------------+

        # Initialize grad
        g_optimizer.zero_grad()
        d_optimizer.zero_grad()

        # make fake images with generator & noise vector 'z'
        z = torch.randn(batch_size, noise_size).to(device)

        """
        FOR CONDITIONAL GAN
        """
        # concat noise vector z with encoded labels
        z_concat = torch.cat((z, label_encoded), 1)
        fake_images = generator(z_concat)
        fake_images_concat = torch.cat((fake_images, label_encoded), 1)

        # Compare result of discriminator with fake images & real labels
        # If generator deceives discriminator, g_loss will decrease
        g_loss = criterion(discriminator(fake_images_concat), real_label)

        # Train generator with backpropagation
        g_loss.backward()
        g_optimizer.step()

        # +---------------------+
        # | train Discriminator |
        # +---------------------+

        # Initialize grad
        d_optimizer.zero_grad()
        g_optimizer.zero_grad()

        # make fake images with generator & noise vector 'z'
        z = torch.randn(batch_size, noise_size).to(device)

        """
        FOR CONDITIONAL GAN
        """
        # concat noise vector z with encoded labels
        z_concat = torch.cat((z, label_encoded), 1)
        fake_images = generator(z_concat)
        fake_images_concat = torch.cat((fake_images, label_encoded), 1)

        # Calculate fake & real loss with generated images above & real images
        fake_loss = criterion(discriminator(fake_images_concat), fake_label)
        real_loss = criterion(discriminator(real_images_concat), real_label)
        d_loss = (fake_loss + real_loss) / 2

        # Train discriminator with backpropagation
        # In this part, we don't train generator
        d_loss.backward()
        d_optimizer.step()

        d_performance = discriminator(real_images_concat).mean()
        g_performance = discriminator(fake_images_concat).mean()

        if (i + 1) % 150 == 0:
            print("Epoch [ {}/{} ]  Step [ {}/{} ]  d_loss : {:.5f}  g_loss : {:.5f}"
                  .format(epoch + 1, num_epoch, i+1, len(data_loader), d_loss.item(), g_loss.item()))

    # print discriminator & generator's performance
    print(" Epock {}'s discriminator performance : {:.2f}  generator performance : {:.2f}"
          .format(epoch + 1, d_performance, g_performance))

    # Save fake images in each epoch
    samples = fake_images.reshape(batch_size, 1, 28, 28)
    save_image(samples, os.path.join(dir_name, 'CGAN_fake_samples{}.png'.format(epoch + 1)))
    # print("label of 'CGAN_fake_samples{}.png' is {}".format(epoch + 1, label))

    # Draw real labels on fake sample images
    # If you got error about this, you can remove lines below
    fake_sample_image = Image.open("{}/CGAN_fake_samples{}.png".format(dir_name, epoch + 1))
    font = ImageFont.truetype("arial.ttf", 17)

    label = label.tolist()
    label = label[:10]
    label = [str(l) for l in label]

    label_text = ", ".join(label)
    label_text = "first 10 labels in this image :\n" + label_text

    image_edit = ImageDraw.Draw(fake_sample_image)
<<<<<<< HEAD
    image_edit.multiline_text(xy=(15, 330),
                              text=label_text,
                              fill=(255, 255, 240),
                              font=font,
                              stroke_width= 2,
                              stroke_fill=(0, 0, 0))
=======
    image_edit.multiline_text((15, 330), label_text, (255, 255, 0), font=font)
>>>>>>> 8ae744a6424e4c3489fe18df7eb3db4d0f46b765
    fake_sample_image.save("{}/CGAN_fake_samples{}.png".format(dir_name, epoch + 1))
