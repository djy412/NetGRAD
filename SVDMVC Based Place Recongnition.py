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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.cluster import KMeans
import torch.nn.functional as F

from dataloading import get_Tokyo_Train_dataloaders_FAST, get_Mapillary_dataloaders, get_Test_Mapillary_dataloaders, get_Tokyo_Train_dataloaders
from config import BATCH_SIZE, WORKERS, LATENT_DIM, EPOCHS, LR, NUM_CLASSES, PATIENCE, BETA, LAMBDA, TOLERANCE, UPDATE_INTERVAL, NUM_VIEWS
from Visualization import Show_settings, Show_dataloader_data, Show_Training_Loss, Show_Component_Embeddings, Show_Componet_Reconstructions, Show_Embedding_Space, Show_Complete_Reconstructions, Show_Partial_Embedding_Space, Show_Results, Show_Representation, Show_NMI_By_Epochs, Show_Variance
from utils import cluster_acc, calculate_purity, set_seed 

Pretrain = True
Evaluate = False

SEED = 5
set_seed(SEED)

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
#----------------------------------------------------------------------------

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
#----------------------------------------------------------------------------

# SDMVC Model
# ------------------------
class SVDMVC(nn.Module):
    def __init__(self, LATENT_DIM, n_clusters ):
        super(SVDMVC, self).__init__()
        self.LATENT_DIM = LATENT_DIM
        self.ae = ResAutoencoder(LATENT_DIM).to(device)
        
        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, LATENT_DIM))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def pretrain(self, data_loader):
        pretrain_ae(self.ae, data_loader)

    def forward(self, x):
        x_hat, z = self.ae(x)
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2))
        q = (q.t() / torch.sum(q, 1)).t()
        return x_hat, z, q
#----------------------------------------------------------------------------

#************************************************************************
#--- Define Sharpening function for target distribution
#************************************************************************
def target_distribution(q):
    weight = q**2 / q.sum(0)
    weight = (weight.t() / weight.sum(1)).t()
    return weight
#----------------------------------------------------------------------------


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
        
        elapsed_time = end_time-start_time
        print(f"\nTraining finished in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        # --- Save Model ---
        torch.save(model.state_dict(), "weights/resautoencoder.pth")
        print("Model saved ✅")
        
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



""" MAIN Program """
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)
    
    print('Loading Mapillary data...')
    train_loader, train_dataset = get_Mapillary_dataloaders(batch_size=BATCH_SIZE, num_workers=WORKERS)
    
    #--- Define the models
    model = SVDMVC(LATENT_DIM, NUM_CLASSES).to(device)
    
    """ --- Pretraining --- """
    model.pretrain(train_loader)
    print("Autoencoder pretrained")
      
    
    print('Loading Tokyo data...')
    train_loader, train_dataset = get_Tokyo_Train_dataloaders(batch_size=BATCH_SIZE, num_workers=WORKERS)
    #train_loader, train_dataset = get_Tokyo_Train_dataloaders_FAST(batch_size=BATCH_SIZE, num_workers=WORKERS, cache_images=True)

    for img1, _, _, _ in train_loader:
        img1 = img1.to(device)
        break
        
    print("Running inference...")
    with torch.no_grad():
        x_hat, _, _= model(img1)
        
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
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    #--- Training Loop ---
    if Pretrain == True:
        print("Starting Training on Tokyo dataset...")
        criterion = nn.MSELoss()
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
        # --- Save Model ---
        torch.save(model.ae.state_dict(), "weights/resautoencoder.pth")
        print("Model saved ✅")
    else: #--- Load models weights
        print("Loading Model wieghts from Pretraining")  
        load_model_path = 'C:/Users/djy41/Desktop/PhD Work/Code/C_2) Visual Place Recognition/weights/resautoencoder.pth'    
        model.ae.load_state_dict(torch.load(load_model_path)) 
        
    print("Running inference...")
    with torch.no_grad():
        x_hat, _, _= model(img1)
        
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
    #--- STEP 1 ---
    print("Initialize cluster centers")
    data=[]
    y_true = []
    for x, _, y, _ in train_loader:
        data.append(x)
        y_true.append(y)
    data = np.concatenate(data)
    y_true = np.concatenate(y_true)
    data = torch.Tensor(data).to(device)
    with torch.no_grad():
        x_hat, z = model.ae(data)
    kmeans = KMeans(n_clusters=NUM_CLASSES, n_init=20)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())   
    #--- Load the initial cluster centers into model
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device) 
    print("Model Cluster centers set")   

    #--- Show the latent space
    Show_Partial_Embedding_Space(z, y_true)    

    # Clear all unneeded data
    x_hat = z = data = None
    torch.cuda.empty_cache()  
    
    data_size = len(train_dataset)
    #--- Test starting accuracy, NMI, and purity
    p = [torch.zeros(data_size, NUM_CLASSES, device=device), torch.zeros(data_size, device=device, dtype=torch.long)]
    y_pred_total, y_true = [],[]  # To collect y_pred y_true across all batches
    for x, _, y, idx in train_loader:  
        x = x.to(device)
        with torch.no_grad():
            _, z, tmp_q = model(x)     
        # update target distribution p
        tmp_q = tmp_q.data
        p[0][idx] = target_distribution(tmp_q)
        p[1][idx] = idx.to(device).long() 
            
        # evaluate clustering performance
        y_pred = tmp_q.cpu().numpy().argmax(1)
        y_pred_total.extend(y_pred)  # Collect y_pred for accuracy, NMI, ARI calculations
                    
        # Collect y_true for accuracy, NMI, ARI calculations
        y_true.extend(y.cpu().numpy())
    
    x = y = tmp_q = p = None #--- clear variables
                
    acc = cluster_acc(y_true, y_pred_total, NUM_CLASSES)
    nmi = nmi_score(y_true, y_pred_total)
    ari = ari_score(y_true, y_pred_total)
    pur = calculate_purity(y_true, y_pred_total)   
    print('Acc {:.4f}'.format(acc),', nmi {:.4f}'.format(nmi), ', purity {:.4f}'.format(pur))
    
    Start_ACC = acc
    Start_NMI = nmi
    Start_PUR = pur    
    
    nmi_view1, nmi_view2, nmi_view3 = [],[],[]
    
    #--- STEP 2 ---
    print("Fine-tuning phase")
    start_time = time.time()
    switch = 1
    NMI_LAST = 0
    previous_value = float('inf')
    stable_count = 0  # Counter for stability
    #--------------------------------------------------------------------------
    for epoch in range(EPOCHS):
        #--- Update the target distribution based on the view with highest probability
        if epoch % UPDATE_INTERVAL == 0:  
            p = [torch.zeros(data_size, NUM_CLASSES, device=device), torch.zeros(data_size, device=device, dtype=torch.long)]
            y_pred_total1, y_pred_total2, y_true = [],[],[]  # To collect y_pred y_true across all batches
                
            for A, P, y, idx in train_loader: 
                A = A.to(device)
                P = P.to(device)
                                
                _, z1, q1 = model(A)
                _, z2, q2 = model(P)
                
                #--- create q based off the variance of q1 and q2
                q1 = q1.detach() #--- gradients not needed
                q2 = q2.detach()

                positions = torch.arange(q1.size(1)).float().to(device)  # Shape: (N,)      
                # Mean for each batch
                mean_q1 = torch.sum(q1 * positions, dim=1, keepdim=True).to(device)  # Shape: (B, 1)
                mean_q2 = torch.sum(q2 * positions, dim=1, keepdim=True).to(device)  # Shape: (B, 1) 
 
                # Variance for each batch
                var_q1 = torch.sum(((positions - mean_q1)**2) * q1, dim=1, keepdim=True).to(device)  
                var_q2 = torch.sum(((positions - mean_q2)**2) * q2, dim=1, keepdim=True).to(device)   
 
                # Weights inversely proportional to variances
                w1 = (1/var_q1) / ((1/var_q1) + (1/var_q2)).to(device) 
                w2 = (1/var_q2) / ((1/var_q1) + (1/var_q2)).to(device)  
 
                # Combine distributions with broadcasting
                q = w1*q1 + w2*q2  # Shape: (B, N)

                # Normalize combined distributions
                q = q / q.sum(dim=1, keepdim=True)    
                    
                # update target distribution p
                p[0][idx] = target_distribution(q)
                p[1][idx] = idx.to(device).long() 
                
                y_pred = q1.cpu().numpy().argmax(1)
                y_pred_total1.extend(y_pred)  #--- Collect y_pred for accuracy, NMI, ARI calculations
                y_pred = q2.cpu().numpy().argmax(1)
                y_pred_total2.extend(y_pred)                       
                y_true.extend(y.cpu().numpy()) #--- Collect y_true 
                       
            #--- Get updated metrics        
            acc = cluster_acc(y_true, y_pred_total1, NUM_CLASSES)
            nmi = nmi_score(y_true, y_pred_total1)
            ari = ari_score(y_true, y_pred_total1)
            print('View1, Iter {}'.format(epoch), ':Acc {:.4f}'.format(acc),', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari))
            nmi_view1.append(nmi) #--- Save the NMI for plotting
            acc = cluster_acc(y_true, y_pred_total2, NUM_CLASSES)
            nmi = nmi_score(y_true, y_pred_total2)
            ari = ari_score(y_true, y_pred_total2)
            print('View2, Iter {}'.format(epoch), ':Acc {:.4f}'.format(acc),', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari))
            nmi_view2.append(nmi) #--- Save the NMI for plotting
            #Show_Partial_Embedding_Space(z1, y)
            #Show_Partial_Embedding_Space(z2, y)
                
            A = P = N = y = idx  = None  # Clear variables for the next iteration
            
            #--- Check for stoping condition
            change = abs(NMI_LAST - nmi)
            print(f"Iteration {epoch}: Change = {change:.5e}")
            if change < TOLERANCE:
                stable_count += 1  # Increment stability counter
                if stable_count >= PATIENCE:  # If stable for `patience` iterations, stop
                    print("Stopping criterion met: Performance measure stabilized.")
                    break
            else:
                stable_count = 0  # Reset counter if change is significant
            NMI_LAST = nmi  # Update previous value            
            
        #--- Train on batches    
        for batch_idx, (x, x2, _, idx) in enumerate(train_loader):
            x = x.to(device)
            x2 = x2.to(device)
            idx = idx.to(device)
              
            x_hat, _, q1 = model(x)                   
            reconstr_loss = F.mse_loss(x_hat, x)
            kl_loss = F.kl_div(q1.log(), p[0][idx], reduction='batchmean')  
            loss1 = BETA*kl_loss + reconstr_loss
            
            x2_hat, _, q2 = model(x2)
            reconstr_loss2 = F.mse_loss(x2_hat, x2)
            kl_loss2 = F.kl_div(q2.log(), p[0][idx], reduction='batchmean')   
            loss2 = BETA*kl_loss2 + reconstr_loss2

            loss = loss1+loss2
         
            optimizer.zero_grad()
            loss.backward()       
            optimizer.step()
    #--------------------------------------------------------------------------    
    end_time = time.time()
    elapsed_time = end_time-start_time
    print(f"\nTraining finished in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

    #--- Save the models
    torch.save(model.state_dict(), './models/SVDMC_Model1.pt')
    print("model saved as {}.".format('SVDMC_Model1.pt'))
    
    #--- Plot the NMI of each view over time
    axis = range(len(nmi_view1))
    plt.figure("NMI")
    plt.plot(axis, nmi_view1, label = 'view1')
    plt.plot(axis, nmi_view2, label = 'view2')
    
    # Labels and legend
    plt.xlabel("Update")
    plt.ylabel("NMI")
    plt.title("Plot of NMI per Update")
    plt.legend()
    plt.grid(True)
    plt.show  
    
    Full_data_a, y_true=[], []
    torch.cuda.empty_cache()
    for a, _, y, _ in train_loader:
        Full_data_a.append(a)
        y_true.append(y)

    Full_data_a = torch.cat(Full_data_a).to(device)
    y_true = torch.cat(y_true)
    
    with torch.no_grad():
        _, z, _ = model(Full_data_a)
    #--- Show the latent space
    Show_Partial_Embedding_Space(z, y_true)
    #Show_Embedding_Space(z, u, y_true)
    
    Full_data_a = y = None # Clear variables 
    
    y_pred_total, y_true = [],[]  # To collect y_pred y_true across all batches
    for _, x, y, idx in train_loader:  # Using shuffled data
        x = x.to(device)
                    
        x_hat, z, tmp_q = model(x)
        # update target distribution p
        tmp_q = tmp_q.data
        p[0][idx] = target_distribution(tmp_q)
        p[1][idx] = idx.to(device).long() 
            
        # evaluate clustering performance
        y_pred = tmp_q.cpu().numpy().argmax(1)
        y_pred_total.extend(y_pred)  # Collect y_pred for accuracy, NMI, ARI calculations
                    
        # Collect y_true for accuracy, NMI, ARI calculations
        y_true.extend(y.cpu().numpy())

    Show_Complete_Reconstructions(x, x_hat)    
                    
    acc = cluster_acc(y_true, y_pred_total, NUM_CLASSES)
    nmi = nmi_score(y_true, y_pred_total)
    ari = ari_score(y_true, y_pred_total)
    pur = calculate_purity(y_true, y_pred_total)   
    print('View 1: Acc {:.4f}'.format(acc),', nmi {:.4f}'.format(nmi), ', purity {:.4f}'.format(pur))
    
    END_ACC = acc
    END_NMI = nmi
    END_PUR = pur
    Show_Results(SEED, Start_ACC, Start_NMI, Start_PUR, END_ACC,  END_NMI, END_PUR)
      
    
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
        


















