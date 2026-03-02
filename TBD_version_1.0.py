# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 20:15:13 2025
Autoencoder for 640x480 images with residual blocks

The goal is to use Deep Multi-view Clustering methods to 
improve view invarient place recognition with added benefit of faster
image retrieval AND a confidence estimate

The main tools being brought to bear are learning a view invariant "place" 
represented as cluster centers through contrastive loss

@author: djy41
"""
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from dataloading import load_data, get_Tokyo_Query_dataloaders, get_Tokyo_Eval_dataloaders, get_Tokyo_Triplet_Train_dataloaders, get_Tokyo_Triplet_Test_dataloaders, get_Mapillary_dataloaders, get_Test_Mapillary_dataloaders, get_Tokyo_Train_dataloaders
from config import BATCH_SIZE, WORKERS, LATENT_DIM, EPOCHS, LR, NUM_CLASSES, PATIENCE, BETA, LAMBDA, TOLERANCE, UPDATE_INTERVAL, NUM_VIEWS
from Visualization import Show_settings, Show_dataloader_data, Show_Training_Loss, Show_Component_Embeddings, Show_Componet_Reconstructions, Show_Embedding_Space, Show_Complete_Reconstructions, Show_Partial_Embedding_Space, Show_Results, Show_Representation, Show_NMI_By_Epochs, Show_Variance
from utils import cluster_acc, calculate_purity, set_seed 

Pretrain = False
Evaluate = True


# Residual Block
# ------------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

# Encoder
# ------------------------
class Encoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super(Encoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),   # 640x480 -> 320x240
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 320x240 -> 160x120
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 160x120 -> 80x60
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),# 80x60 -> 40x30
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            ResidualBlock(256),
        )

        self.flatten_dim = 256 * 40 * 30 # 640x480
        self.fc = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        z = self.fc(out)
        return z

# Decoder
# ------------------------
class Decoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super(Decoder, self).__init__()

        self.unflatten_dim = (256, 30, 40) # 640x480
        self.fc = nn.Linear(latent_dim, 256 * 40 * 30)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 40x30 -> 80x60
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 80x60 -> 160x120
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 160x120 -> 320x240
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResidualBlock(32),
            
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),    # 320x240 -> 640x480
            nn.Sigmoid()
        )

    def forward(self, z):
        out = self.fc(z)
        out = out.view(out.size(0), *self.unflatten_dim)  # reshape to (256,40,30)
        out = self.deconv(out)
        return out

# Full Autoencoder
# ------------------------
class ResAutoencoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super(ResAutoencoder, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out, z

""" MAIN Program """
if __name__ == "__main__":
    
    if Pretrain == True:
        print('Loading Training data...')
        train_loader, train_dataset = get_Mapillary_dataloaders(batch_size=BATCH_SIZE, num_workers=WORKERS)
    
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
        model = ResAutoencoder(latent_dim=LATENT_DIM).to(device)
        #--- Loss + Optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
        #--- Training Loop ---
        print("Starting Training...")
        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0
        
            for imgs in train_loader:
                imgs = imgs.to(device)
        
                #--- Forward
                outputs, _ = model(imgs)
                loss = criterion(outputs, imgs)  # reconstruction loss
        
                #--- Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
                running_loss += loss.item() * imgs.size(0)
        
            epoch_loss = running_loss / len(train_dataset)
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss:.6f}")   
        
        # --- Save Model ---
        torch.save(model.state_dict(), "resautoencoder.pth")
        print("Model saved ✅")
        
        # Clear PyTorch cache
        torch.cuda.empty_cache()
        
        
    if Evaluate == True:
        print('Loading Testing data...')
        test_loader, test_dataset = get_Mapillary_dataloaders(batch_size=BATCH_SIZE, num_workers=WORKERS)
    
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
        model = ResAutoencoder(latent_dim=LATENT_DIM).to(device)
        
        load_model_path = 'C:/Users/djy41/Desktop/PhD Work/Code/C_2) Visual Place Recognition/weights/resautoencoder.pth'    
        model.load_state_dict(torch.load(load_model_path)) 
        
        for imgs in test_loader:
            imgs = imgs.to(device)
            break
        
        print("Running inference...")
        x_hat, _ = model(imgs)
        
        x_hat = x_hat.detach()
        #--- Show an orignial and reconstructed image
        plt.figure()
        plt.title(f"{EPOCHS} Epoch")
        plt.imshow(imgs[0].squeeze().permute(1,2,0).cpu())
        plt.show()
        plt.figure()
        plt.title(f"Latent Dims:{LATENT_DIM}")
        plt.imshow(x_hat[0].squeeze().permute(1,2,0).cpu())
        plt.show()
        
        # Clear PyTorch cache
        torch.cuda.empty_cache()
        

















