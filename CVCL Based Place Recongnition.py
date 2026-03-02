# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 12:51:24 2025
SVDMVC for place representation and loop closure detection

Workflow: 
    1) Train AE network on Environment or similar Environment
    2) Build a database of representations (discriptors) 
    3) Run unseen query images against the database measuring similarity
    4) Return top N matches
    5) Evaluate matches using Recall@N, whereby a query image is correctly 
localized if at least one of the top N images is within the ground truth tolerance (25m?)
This method is for 640x480 images. 

@author: djy41
"""
import torch
import time
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.nn.parameter import Parameter

from dataloading import get_Mapillary_dataloaders, get_Test_Mapillary_dataloaders, get_Tokyo_Train_dataloaders
from config import BATCH_SIZE, WORKERS, LATENT_DIM, EPOCHS, LR, NUM_CLASSES, BETA, LAMBDA, TEMP, NORM, NUM_VIEWS
from Visualization import Show_settings, Show_dataloader_data, Show_Training_Loss, Show_Component_Embeddings, Show_Componet_Reconstructions, Show_Embedding_Space, Show_Complete_Reconstructions, Show_Partial_Embedding_Space, Show_Results, Show_Representation, Show_NMI_By_Epochs, Show_Variance
from loss import DeepMVCLoss

Pretrain = False
Evaluate = False

dim_high_feature = LATENT_DIM
dim_low_feature = 514


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

            # ✅ Keep only here
            ResidualBlock(256),
        )

        self.flatten_dim = 256 * 40 * 30 # 640x480
        self.fc = nn.Linear(self.flatten_dim, latent_dim)
        
        #self.flatten_dim = 256 * 15 * 20 # 320x240
        #self.fc = nn.Linear(self.flatten_dim, latent_dim)

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
        
        #self.unflatten_dim = (256, 15, 20) # 320x240
        #self.fc = nn.Linear(latent_dim, 256 * 15 * 20)

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
            ResidualBlock(32),   # ✅ keep only one residual block here
            
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
        return out,z

# CVCL Model
# ------------------------
class CVCLNetwork(nn.Module):
    def __init__(self, dim_high_feature, dim_low_feature, num_clusters, LATENT_DIM):
        super(CVCLNetwork, self).__init__()
        self.LATENT_DIM = LATENT_DIM
        self.ae = ResAutoencoder(LATENT_DIM)

        self.label_learning_module = nn.Sequential(
            nn.Linear(dim_high_feature, dim_low_feature),
            nn.Linear(dim_low_feature, num_clusters),
            nn.Softmax(dim=1)
        )
        
    def pretrain(self, data_loader):
        pretrain_ae(self.ae, data_loader)

    def forward(self, x):
        x_hat, z = self.ae(x)
        label_probs = self.label_learning_module(z)

        return label_probs, x_hat, z

#--- Define Autoencoder pre-training
#************************************************************************
def pretrain_ae(model, data_loader): #- Takes in the model, the data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    if Pretrain == True:
        print("Pre-training Autoencoder")
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LR)
            
        Loss_histroy = []     
        print("Starting Training...")
        start_time = time.time()
        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0
        
            for imgs in data_loader:
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
            Loss_histroy.append(epoch_loss)
        end_time = time.time()    
        #--- Plot the training loss
        Show_Training_Loss(Loss_histroy)
        
        # --- Save Model ---
        torch.save(model.state_dict(), "weights/resautoencoder.pth")
        print("Model saved ✅")
        elapsed_time = end_time-start_time
        print(f"\nTraining finished in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        
    else: #--- Load models weights
        print("Loading Model wieghts from Pretraining")  
        load_model_path = 'C:/Users/djy41/Desktop/PhD Work/Code/C_2) Visual Place Recognition/weights/resautoencoder.pth'    
        model.load_state_dict(torch.load(load_model_path))     
    
    for imgs in data_loader:
        imgs = imgs.to(device)
        break
    print("Running inference...")
    with torch.no_grad():
        x_hat, _ = model(imgs)
    Show_Complete_Reconstructions(imgs, x_hat)    

#----------------------------------------------------------------------------

#--- Define Model Train
#************************************************************************
def contrastive_train(network_model, train_loader, train_dataset, mvc_loss, batch_size, LAMBDA, BETA, TEMP, NORM, epoch, optimizer, num_views):
    network_model.train()
    criterion = torch.nn.MSELoss()
    total_loss = 0.0

    for batch_idx, (imgs1, imgs2, _, _) in enumerate(train_loader):
        imgs1 = imgs1.to(device)
        imgs2 = imgs2.to(device)            
        label_probs1, x_hat1, _ = network_model(imgs1)
        label_probs2, x_hat2, _ = network_model(imgs2)
        
        loss_list = list()

        loss_list.append(LAMBDA * mvc_loss.forward_label(label_probs1, label_probs2, TEMP, NORM))
        loss_list.append(BETA * mvc_loss.forward_prob(label_probs1, label_probs2))
        loss_list.append(criterion(imgs1, x_hat1))
        loss_list.append(criterion(imgs2, x_hat2))
        loss = sum(loss_list)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    print(label_probs1)
    epoch_loss = total_loss / len(train_dataset)
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss:.6f}")  

    return total_loss
#----------------------------------------------------------------------------


""" MAIN Program """
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)
    
    print('Loading Mapillary data...')
    train_loader, train_dataset = get_Mapillary_dataloaders(batch_size=BATCH_SIZE, num_workers=WORKERS)
    
    #--- Define the models
    model = CVCLNetwork(dim_high_feature, dim_low_feature, NUM_CLASSES, LATENT_DIM).to(device)
    #--- Define the loss
    mvc_loss = DeepMVCLoss(BATCH_SIZE, NUM_CLASSES)
    
    """ --- Pretraining --- """
    model.pretrain(train_loader)
    print("Autoencoder pretrained")
      
    
    print('Loading Tokyo data...')
    train_loader, train_dataset = get_Tokyo_Train_dataloaders(batch_size=BATCH_SIZE, num_workers=WORKERS)

    for img1, _, _, _ in train_loader:
        img1 = img1.to(device)
        break
        
    print("Running inference...")
    with torch.no_grad():
        _, x_hat, _= model(img1)
        
    x_hat = x_hat.detach()
    #--- Show an orignial and reconstructed image
    plt.figure(1)
    plt.title("Tokyo")
    plt.imshow(img1[0].squeeze().permute(1,2,0).cpu())
    plt.show()
    plt.figure(2)
    plt.title("Reconstruction")
    plt.imshow(x_hat[0].squeeze().permute(1,2,0).cpu())
    plt.show()
    
    
    #--- Training Loop ---
    print("Starting Training on Tokyo dataset...")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    start_time = time.time()
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
    
        for imgs1, _, _, _ in train_loader:
            imgs1 = imgs1.to(device)
            #imgs2 = imgs2.to(device)

            #--- Forward
            outputs, _ = model.ae(imgs1)
            loss = criterion(outputs, imgs1)  # reconstruction loss
            #outputs, _ = model.ae(imgs2)
            #loss2 = criterion(outputs, imgs2)  # reconstruction loss
            #loss = torch.mean(loss1+loss2)
        
            #--- Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item() * imgs1.size(0)
        
        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss:.6f}")       
    end_time = time.time()
    elapsed_time = end_time-start_time
    print(f"\nTraining finished in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    
    print("Running inference...")
    with torch.no_grad():
        _, x_hat, _= model(img1)
        
    x_hat = x_hat.detach()
    #--- Show an orignial and reconstructed image
    plt.figure(3)
    plt.title("Tokyo")
    plt.imshow(img1[0].squeeze().permute(1,2,0).cpu())
    plt.show()
    plt.figure(4)
    plt.title("Reconstruction")
    plt.imshow(x_hat[0].squeeze().permute(1,2,0).cpu())
    plt.show()
    
    
    
    """ --- Clustering --- """ 
    print("Starting Contrastive Training")
    t = time.time()
    fine_tuning_loss_values = np.zeros(EPOCHS, dtype=np.float64)
    for epoch in range(EPOCHS):
        total_loss = contrastive_train(model, train_loader, train_dataset, mvc_loss, BATCH_SIZE, LAMBDA, BETA, TEMP, NORM, epoch, optimizer, NUM_VIEWS)
        fine_tuning_loss_values[epoch] = total_loss
        
    print("contrastive_train finished.")
    print("Total time elapsed: {:.2f}s".format(time.time() - t))

    torch.save(model.state_dict(), './models/CVCL_pytorch_model.pth' )

    
    # #--- Clear PyTorch cache
    # torch.cuda.empty_cache()
    
    
    """ --- Evaluation --- """
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
        x_hat, _ , _ = model(imgs)
        
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
        


















