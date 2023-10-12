# Import torch, os, and dgl
import torch
import torch.nn as nn
import torch.optim as optim
import os
import dgl
import pickle

# Import generator and discriminator networks
from generator import Generator
from discriminator import Discriminator

# Define the loss functions (e.g., BCE loss)
criterion = nn.BCEWithLogitsLoss()

# Initialize generator and discriminator instances
input_dim = 100  # Replace with your input dimension

# Load the etypes from pickle
with open('rel_names.pkl', 'rb') as f:
    rel_names = pickle.load(f)

# Initialize generator and discriminator instances
generator = Generator(input_dim, rel_names)
discriminator = Discriminator(rel_names)

# Define optimizers for both networks
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Get graphs
folder = 'dgl_graphs'
files = os.listdir(folder)

# Training loop
num_epochs = 20  # Update with your desired number of epochs
for epoch in range(num_epochs):
    print("EPOCH:", epoch)
    for file in files:
        # Load graph
        graph = dgl.load_graphs(os.path.join(folder, file))[0][0]

        # Train the discriminator
        optimizer_D.zero_grad()

        # Generate real labels (1 for real)
        real_labels = torch.ones(1, 1)  # Assuming batch size of 1
        
        # Forward pass through the discriminator to get predictions
        discriminator_outputs_real = discriminator(graph)
        
        # Calculate the discriminator loss for real graphs
        discriminator_loss_real = criterion(discriminator_outputs_real, real_labels)
        
        # Backpropagate and update discriminator weights
        discriminator_loss_real.backward()
        optimizer_D.step()
        
        # Train the generator
        optimizer_G.zero_grad()
        
        # Generate random input noise
        input_dim = 100  # Example: 100-dimensional input noise
        batch_size = 1  # Example: batch size of 1
        input_data = torch.randn(batch_size, input_dim).to(device)
        
        # Generate fake graphs from input noise using the generator
        fake_graphs = generator(input_data)
        
        # Generate fake labels (0 for fake)
        fake_labels = torch.zeros(1, 1)  # Assuming batch size of 1
        
        # Forward pass through the discriminator to get predictions for fake graphs
        discriminator_outputs_fake = discriminator(fake_graphs)
        
        # Calculate the generator loss
        generator_loss = criterion(discriminator_outputs_fake, fake_labels)
        
        # Backpropagate and update generator weights
        generator_loss.backward()
        optimizer_G.step()

    # Print and save intermediate results
    print("Discriminator loss:", discriminator_loss_real.item())
    print("Generator loss:", generator_loss.item())
    print("")
    torch.save(generator.state_dict(), "generator.pth")
    torch.save(discriminator.state_dict(), "discriminator.pth")

# Generate floorplan graphs/images using the trained generator

