import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, datasets
import pytorch_lightning as pl

class Generator(nn.Module):
  '''
  Generator class in a CGAN. Accepts a noise tensor (latent dim 100)
  and a label tensor as input as outputs another tensor of size 784.
  Objective is to generate an output tensor that is indistinguishable 
  from the real MNIST digits.
  '''

  def __init__(self):
    super().__init__()
    self.embedding = nn.Embedding(10, 10)
    self.layer1 = nn.Sequential(nn.Linear(in_features=100+10, out_features=256),
                                nn.LeakyReLU())
    self.layer2 = nn.Sequential(nn.Linear(in_features=256, out_features=512),
                                nn.LeakyReLU())
    self.layer3 = nn.Sequential(nn.Linear(in_features=512, out_features=1024),
                                nn.LeakyReLU())
    self.output = nn.Sequential(nn.Linear(in_features=1024, out_features=28*28),
                                nn.Tanh())

  def forward(self, z, y):
    # pass the labels into a embedding layer
    labels_embedding = self.embedding(y)
    # concat the embedded labels and the noise tensor
    # x is a tensor of size (batch_size, 110)
    x = torch.cat([z, labels_embedding], dim=-1)    
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.output(x)
    return x


class Discriminator(nn.Module):
  '''
  Discriminator class in a CGAN. Accepts a tensor of size 784 and
  a label tensor as input and outputs a tensor of size 1,
  with the predicted class probabilities (generated or real data)
  '''

  def __init__(self):
    super().__init__()
    self.embedding = nn.Embedding(10, 10)
    self.layer1 = nn.Sequential(nn.Linear(in_features=28*28+10, out_features=1024),
                                nn.LeakyReLU())
    self.layer2 = nn.Sequential(nn.Linear(in_features=1024, out_features=512),
                                nn.LeakyReLU())
    self.layer3 = nn.Sequential(nn.Linear(in_features=512, out_features=256),
                                nn.LeakyReLU())
    self.output = nn.Sequential(nn.Linear(in_features=256, out_features=1),
                                nn.Sigmoid())
    
  def forward(self, x, y):
    # pass the labels into a embedding layer
    labels_embedding = self.embedding(y)
    # concat the embedded labels and the input tensor
    # x is a tensor of size (batch_size, 794)
    x = torch.cat([x, labels_embedding], dim=-1)    
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.output(x)
    return x


class CGAN(pl.LightningModule):

  def __init__(self):
    super().__init__()
    self.generator = Generator()
    self.discriminator = Discriminator()

  def forward(self, z, y):
    """
    Generates an image using the generator
    given input noise z and labels y
    """
    return self.generator(z, y)

  def generator_step(self, x):
    """
    Training step for generator
    1. Sample random noise and labels
    2. Pass noise and labels to generator to
       generate images
    3. Classify generated images using
       the discriminator
    4. Backprop loss
    """
    
    # Sample random noise and labels
    z = torch.randn(x.shape[0], 100, device=device)
    y = torch.randint(0, 10, size=(x.shape[0],), device=device)

    # Generate images
    generated_imgs = self(z, y)

    # Classify generated image using the discriminator
    d_output = torch.squeeze(self.discriminator(generated_imgs, y))

    # Backprop loss. We want to maximize the discriminator's
    # loss, which is equivalent to minimizing the loss with the true
    # labels flipped (i.e. y_true=1 for fake images). We do this
    # as PyTorch can only minimize a function instead of maximizing
    g_loss = nn.BCELoss()(d_output,
                           torch.ones(x.shape[0], device=device))

    return g_loss

  def discriminator_step(self, x, y):
    """
    Training step for discriminator
    1. Get actual images and labels
    2. Predict probabilities of actual images and get BCE loss
    3. Get fake images from generator
    4. Predict probabilities of fake images and get BCE loss
    5. Combine loss from both and backprop
    """
    
    # Real images
    d_output = torch.squeeze(self.discriminator(x, y))
    loss_real = nn.BCELoss()(d_output,
                             torch.ones(x.shape[0], device=device))

    # Fake images
    z = torch.randn(x.shape[0], 100, device=device)
    y = torch.randint(0, 10, size=(x.shape[0],), device=device)

    generated_imgs = self(z, y)
    d_output = torch.squeeze(self.discriminator(generated_imgs, y))
    loss_fake = nn.BCELoss()(d_output,
                             torch.zeros(x.shape[0], device=device))

    return loss_real + loss_fake

  def training_step(self, batch, batch_idx, optimizer_idx):
    X, y = batch

    # train generator
    if optimizer_idx == 0:
      loss = self.generator_step(X)
    
    # train discriminator
    if optimizer_idx == 1:
      loss = self.discriminator_step(X, y)

    return loss

  def configure_optimizers(self):
    g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.0002)
    d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002)
    return [g_optimizer, d_optimizer], []


if __name__ == "__main__":
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  mnist_transforms = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.5], std=[0.5]),
                                      transforms.Lambda(lambda x: x.view(-1, 784)),
                                      transforms.Lambda(lambda x: torch.squeeze(x))
                                      ])

  data = datasets.MNIST(root='../data/MNIST', download=True, transform=mnist_transforms)

  mnist_dataloader = DataLoader(data, batch_size=128, shuffle=True, num_workers=0) 

  model = CGAN()

  trainer = pl.Trainer(max_epochs=100, gpus=1 if torch.cuda.is_available() else 0, progress_bar_refresh_rate=50)
  trainer.fit(model, mnist_dataloader)
