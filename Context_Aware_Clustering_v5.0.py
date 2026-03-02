# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 11:03:32 2025

@author: djy41
"""
import os
import psutil
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

from dataloading import get_RobotCar_Seq_Eval_db_dataloaders, get_RobotCar_25_Train_dataloaders, get_RobotCar_25_Triplet_Train_dataloaders, get_RobotCar_25_Eval_query_dataloaders, get_RobotCar_25_Eval_db_dataloaders, get_Pittsburg30k_Eval_query_dataloaders, get_Pittsburg30k_Eval_db_dataloaders, get_Pittsburg30k_Train_dataloaders, get_Pittsburg30k_Triplet_Train_dataloaders ,get_Map_Tokyo_dataloaders, load_data, get_Tokyo_Query_dataloaders, get_Tokyo_Eval_dataloaders, get_Tokyo_Triplet_Train_dataloaders, get_Tokyo_Triplet_Test_dataloaders, get_Mapillary_dataloaders, get_Test_Mapillary_dataloaders, get_Tokyo_Train_dataloaders
from config import EMA_TAU, EPS, MU_DIV, KAPPA_END, KAPPA_START, CHANNELS, dataset_name, MARGIN, NUM_FEATURE_CLUSTERS, BATCH_SIZE, WORKERS, LATENT_DIM, EPOCHS, LR, NUM_CLASSES, PATIENCE, BETA, LAMBDA, TOLERANCE, UPDATE_INTERVAL, NUM_VIEWS
from Visualization import Show_settings, Show_dataloader_data, Show_Training_Loss, Show_Component_Embeddings, Show_Componet_Reconstructions, Show_Embedding_Space, Show_Complete_Reconstructions, Show_Partial_Embedding_Space, Show_Results, Show_Representation, Show_NMI_By_Epochs, Show_Variance
from utils import compute_embeddings, retrieve_top_k, Compute_recall_at_N, batch_to_emb, streaming_topk_for_query, visualize_streaming_topk, to_numpy_image, draw_with_border, OnlinePlaceMemory, pick_triplet, show_pair, get_image_by_index, cluster_acc, calculate_purity, set_seed 
from collections import deque
import collections

Pretrain = False
Load_Model = False
Evaluate = True
Evaluate_on_Image_Stream = True

SEED = 1
set_seed(SEED)

# Residual Block
# ------------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels, groups=16):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.gn1   = nn.GroupNorm(min(groups, channels), channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.gn2   = nn.GroupNorm(min(groups, channels), channels)

    def forward(self, x):
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out = out + x
        return self.relu(out)
#----------------------------------------------------------------------------

# Generalized Mean Pooling Block
# ------------------------
class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        # p can be shared across channels or per-channel (vector) if you want more flexibility
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):  # x: [B, C, H, W]
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.avg_pool2d(x, (x.size(-2), x.size(-1)))  # global average pooling
        x = x.pow(1.0 / self.p)
        return x.view(x.size(0), -1)  # [B, C]

# Encoder
# ------------------------
# --- Encoder: return multi-scale features + top feat + z ---
class Encoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.s1 = nn.Sequential(
            nn.Conv2d(CHANNELS, 32, 3, stride=2, padding=1),
            nn.GroupNorm(8, 32), nn.ReLU(inplace=True))
        self.s2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64), nn.ReLU(inplace=True))
        self.s3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(16, 128), nn.ReLU(inplace=True))
        self.s4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.GroupNorm(16, 256), nn.ReLU(inplace=True))
        self.s5 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.GroupNorm(32, 512), nn.ReLU(inplace=True),
            ResidualBlock(512))
        self.pool = GeM()
        self.fc   = nn.Linear(512, latent_dim)

    def forward(self, x):
        f1 = self.s1(x)        # [B,  32, 240, 320]
        f2 = self.s2(f1)       # [B,  64, 120, 160]
        f3 = self.s3(f2)       # [B, 128,  60,  80]
        f4 = self.s4(f3)       # [B, 256,  30,  40]
        f5 = self.s5(f4)       # [B, 512,  15,  20]
        z  = self.fc(self.pool(f5))  # [B, D]
        return (f1, f2, f3, f4, f5), z
#----------------------------------------------------------------------------
   
# Decoder
# ------------------------
# --- Decoder: take top feature map + skip connections ---
class Decoder(nn.Module):
    def __init__(self, out_ch=CHANNELS):
        super().__init__()
        # up from 512x15x20 → ... → out
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 15x20 -> 30x40
            nn.GroupNorm(16, 256), nn.ReLU(inplace=True))
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256+256, 128, 4, 2, 1),  # concat with f4 (256)
            nn.GroupNorm(16, 128), nn.ReLU(inplace=True))
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128+128, 64, 4, 2, 1),   # concat f3 (128)
            nn.GroupNorm(8, 64), nn.ReLU(inplace=True))
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(64+64, 32, 4, 2, 1),     # concat f2 (64)
            nn.GroupNorm(8, 32), nn.ReLU(inplace=True),
            ResidualBlock(32))
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(32+32, out_ch, 4, 2, 1)) # concat f1 (32)
        # Final activation will depend on input scaling (see below)

    def forward(self, feats):
        f1, f2, f3, f4, f5 = feats
        x = self.up1(f5)                  # 30x40
        x = torch.cat([x, f4], dim=1)     # 30x40
        x = self.up2(x)                   # 60x80
        x = torch.cat([x, f3], dim=1)
        x = self.up3(x)                   # 120x160
        x = torch.cat([x, f2], dim=1)
        x = self.up4(x)                   # 240x320
        x = torch.cat([x, f1], dim=1)
        x = self.up5(x)                   # 480x640
        return x
#----------------------------------------------------------------------------

# Full Autoencoder
# ------------------------
# --- Autoencoder wrapper remains, but pass feature maps to decoder ---
class ResAutoencoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(out_ch=CHANNELS)

    def forward(self, x):
        feats, z = self.encoder(x)
        recon_logits = self.decoder(feats)   # no activation yet (see loss below)
        return recon_logits, z
#----------------------------------------------------------------------------
    
# Context aware Model
# ------------------------
class ContextClusteringModel(nn.Module):
    def __init__(self, latent_dim, num_feature_clusters, num_classes, normalize_inputs=True):
        super().__init__()
        self.ae = ResAutoencoder(latent_dim)
        self.latent_dim = latent_dim
        self.normalize_inputs = normalize_inputs
        self.K = num_feature_clusters

        self.context_centers = nn.Parameter(torch.Tensor(self.K, latent_dim))
        torch.nn.init.xavier_normal_(self.context_centers.data)

        self.kappa = KAPPA_START
        self.eps = EPS

    def pretrain(self, data_loader):
        pretrain_ae(self.ae, data_loader)

    def forward(self, x):
        x_hat, z = self.ae(x)
        if self.normalize_inputs:
            z_norm = F.normalize(z, p=2, dim=1, eps=self.eps)
            centers = F.normalize(self.context_centers, p=2, dim=1, eps=self.eps)
        else:
            z_norm, centers = z, self.context_centers

        # squared distances
        z2 = (z_norm**2).sum(dim=1, keepdim=True)        # [B,1]
        c2 = (centers**2).sum(dim=1).unsqueeze(0)        # [1,K]
        cross = z_norm @ centers.t()                     # [B,K]
        sqdist = (z2 - 2*cross + c2).clamp_min(0.0)      # [B,K]

        # FIX: softmax over negative distances (with sharpness kappa)
        logits = -self.kappa * sqdist
        a = F.softmax(logits, dim=1)                     # [B,K]

        weighted_centers = a @ centers                   # [B,D]
        residual = z_norm - weighted_centers             # [B,D]
        z_ctx = F.normalize(residual, p=2, dim=1, eps=self.eps)

        # we return centers (unit) for diversity loss
        return x_hat, z_ctx, {"centers_norm": centers}

    @staticmethod
    def diversity_loss(centers_norm: torch.Tensor):
        """
        Penalize similarity between different prototypes.
        L_div = mean_{i!=j} (cos(m_i, m_j))^2
        """
        K = centers_norm.shape[0]
        if K <= 1:
            return centers_norm.new_tensor(0.0)
        gram = centers_norm @ centers_norm.t()           # [K,K], cos-sim since unit-norm
        off = gram - torch.diag(torch.diag(gram))
        return (off**2).sum() / (K*(K-1) + 1e-8)

    @torch.no_grad()
    def ema_update_centers(self, z_norm, a, tau: float = EMA_TAU):
        if tau <= 0.0:  # keep disabled to honor "single regularizer"
            return
        weights = a.sum(dim=0, keepdim=True) + self.eps
        soft_means = (a.t() @ z_norm) / weights.t()
        soft_means = F.normalize(soft_means, p=2, dim=1, eps=self.eps)
        self.context_centers.mul_(1.0 - tau).add_(tau * soft_means)
#----------------------------------------------------------------------------

#************************************************************************
#--- Define Triplet Loss
#************************************************************************
def Triplet_Loss(z_a_batch, z_p_batch, z_n_batch, margin = MARGIN):
    distance_ap = torch.norm(z_a_batch - z_p_batch, p=2, dim=1)
    distance_an = torch.norm(z_a_batch - z_n_batch, p=2, dim=1)
            
    loss = torch.mean(torch.relu(distance_ap - distance_an + margin)) 

    return loss
#----------------------------------------------------------------------------

#************************************************************************
#--- Define Triplet Loss
#************************************************************************
def triplet_loss(anchor, positive, negatives, margin=0.5):
    # anchor: [D], positive: [D], negatives: [K,D]; all unit-norm
    # cosine distance: d = 1 - cos
    ap = 1.0 - (anchor @ positive)          # scalar
    an = 1.0 - (anchor @ negatives.t())     # [K]
    # semi-hard: only negatives closer than margin boundary
    losses = torch.relu(ap - an + margin)   # [K]
    
    return losses.mean()
#----------------------------------------------------------------------------

#--- Define Autoencoder pre-training
#************************************************************************
def pretrain_ae(model, data_loader): #- Takes in the model, the data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    if Pretrain:
        print("Pre-training Autoencoder")
        optimizer = optim.Adam(model.parameters(), lr=LR)
              
        print("Starting Training...")
        start_time = time.time()
        for epoch in range(10):
            model.train()
            running_loss = 0.0
        
            for img, _ in data_loader:
                img = img.to(device)
             
                #--- Forward
                recon_logits, _ = model(img)
                loss = F.binary_cross_entropy_with_logits(recon_logits, img)

                #--- Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
                running_loss += loss.item() * img.size(0)
            
            epoch_loss = running_loss / len(train_dataset)
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss:.6f}") 
            
        end_time = time.time()            
        elapsed_time = end_time-start_time
        print(f"\nTraining finished in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        # --- Save Model ---
        torch.save(model.state_dict(), "weights/Pretrain_CA_resautoencoder.pth")
        print("Model saved ✅")
        del optimizer
        torch.cuda.empty_cache()
    
    else: #--- Load models weights
        print("Loading Model wieghts from Pretraining")  
        load_model_path = 'C:/Users/djy41/Desktop/PhD Work/Code/C_2) Visual Place Recognition/weights/Pretrain_CA_resautoencoder.pth'    
        model.load_state_dict(torch.load(load_model_path))  
#----------------------------------------------------------------------------


# Create process handle once outside loop
process = psutil.Process(os.getpid())
memory_usage = []   # store memory in MB
memory_len = []     # store number of memory items


""" MAIN Program """
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)
       
    """--- Show the Run Settings ---"""
    Show_settings()
    
    #--- Define the models
    model = ContextClusteringModel(LATENT_DIM, NUM_FEATURE_CLUSTERS, NUM_CLASSES, True).to(device)
   
    # print('Loading Map Tokyo data...')
    # train_loader, train_dataset = get_Map_Tokyo_dataloaders(batch_size=BATCH_SIZE, num_workers=WORKERS)
    
    #print('Loading Pittsburg30k data...')
    #train_loader, train_dataset = get_Pittsburg30k_Train_dataloaders(batch_size=BATCH_SIZE, num_workers=WORKERS)

    print('Loading RobotCar data...')
    train_loader, train_dataset = get_RobotCar_25_Train_dataloaders(batch_size=BATCH_SIZE, num_workers=WORKERS)

    #train_dataset, dims, view, data_size, class_num = load_data("MULTI-STL-10")
    #train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,shuffle=True, drop_last=True,)      
    
    # """ --- Pretraining --- """
    print("Starting Training on dataset...")
    model.pretrain(train_loader)
    print("Autoencoder pretrained")

    """--- Now initialize cluster centers on the 64 most common features  ---"""
    z = []
    for img, _ in train_loader: 
        img = img.to(device)

        with torch.no_grad():
            _, z_out = model.ae(img)
        z_out = F.normalize(z_out, p=2, dim=1)
        z.append(z_out)
    
    # Concatenate along the batch dimension
    z = torch.cat(z, dim=0)   # shape [N, latent_dim]
    latents = z.cpu().numpy()  # convert to numpy for sklearn KMeans
 
    print("Running Kmeans")
    kmeans = KMeans(n_clusters=NUM_FEATURE_CLUSTERS, random_state=SEED, n_init=20, max_iter=200)
    kmeans.fit(latents)
    #--- Convert centers to torch tensor and initialize model centers on normalized tensors
    C = torch.tensor(kmeans.cluster_centers_, dtype=torch.float).to(device)
    with torch.no_grad():
        model.context_centers.copy_(F.normalize(C, p=2, dim=1))
   
    # print('Loading Tokyo Triplet dataset...')
    # train_loader, train_dataset = get_Tokyo_Triplet_Train_dataloaders(batch_size=BATCH_SIZE, num_workers=WORKERS)

    #print('Loading Pittsburg30k Triplet dataset...')
    #train_loader, train_dataset = get_Pittsburg30k_Triplet_Train_dataloaders(batch_size=BATCH_SIZE, num_workers=WORKERS)

    print('Loading Triplet dataset...')
    train_loader, train_dataset = get_RobotCar_25_Triplet_Train_dataloaders(batch_size=BATCH_SIZE, num_workers=WORKERS)

    for A, _, _, y, _ in train_loader: 
        print("Anchor batch:", A.shape)
        print("Label batch:", y.shape)
        A = A.to(device)
        y = y
        break
    
    print("Running inference...")
    with torch.no_grad():
        A_out, z, _ = model(A)
        A_out = torch.sigmoid(A_out)
    
    #--- Show an orignial and reconstructed image 
    plt.figure()
    plt.title(dataset_name)
    if CHANNELS == 3:
        plt.imshow(A[0].squeeze().permute(1,2,0).cpu())
    else:
        plt.imshow(A[0].squeeze().cpu())    
    plt.show()
    plt.figure()
    plt.title("Reconstruction")
    if CHANNELS == 3:
        plt.imshow(A_out[0].squeeze().permute(1,2,0).detach().cpu())
    else:
        plt.imshow(A_out[0].squeeze().detach().cpu())
    plt.show()
    #--- Show initial embedded space
    #Show_Partial_Embedding_Space(z, y)
  

    if Load_Model==False:
        # """--- Train Context Model to pull out the 64 common features of an embedding ---"""
        print("Triplet Training Autoencoder")
        bce_logits = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=LR)
                
        Loss_histroy = []     
        print("Starting Training...")
        start_time = time.time()
        for epoch in range(EPOCHS):
            model.train()
            
            # Anneal assignment sharpness
            if EPOCHS > 1:
                t = min(1.0, epoch / max(1, EPOCHS // 3))
                model.kappa = KAPPA_START * (1.0 - t) + KAPPA_END * t
            
            running_loss = 0.0
            for A, P, N, Y, _ in train_loader:
                A = A.to(device); P = P.to(device); N = N.to(device)
        
                A_out, z_A, ctxA = model(A)
                P_out, z_P, ctxP = model(P)
                N_out, z_N, ctxN = model(N)
        
                loss_rec = (bce_logits(A_out, A) + bce_logits(P_out, P) + bce_logits(N_out, N))
                loss_tri = Triplet_Loss(z_A, z_P, z_N, margin=MARGIN)
        
                #--- SINGLE regularizer: prototype diversity
                L_div = model.diversity_loss(ctxA["centers_norm"])
        
                loss = loss_rec + LAMBDA * loss_tri + MU_DIV * L_div
        
                #--- Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
                running_loss += loss.item() * A.size(0)
                
            epoch_loss = running_loss / len(train_dataset)
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss:.6f}") 
            Loss_histroy.append(running_loss)
                
        end_time = time.time()            
        elapsed_time = end_time-start_time
        print(f"\nTraining finished in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        # --- Save Model ---
        torch.save(model.state_dict(), "weights/CA_model.pth")
        A_out = torch.sigmoid(A_out) #--- For showing image results        
        print("Model saved ✅")
        del optimizer
        torch.cuda.empty_cache()
        Show_Training_Loss(Loss_histroy)
        
    else:
        for A, _, _, Y, _ in train_loader:
            A=A.to(device)
            with torch.no_grad():
                #--- Forward
                A_out, z_A, _ = model(A)
                A_out = torch.sigmoid(A_out)
            break
        print("Loading Model weights from memory")  
        load_model_path = 'C:/Users/djy41/Desktop/PhD Work/Code/C_2) Visual Place Recognition/weights/CA_model.pth'    
        model.load_state_dict(torch.load(load_model_path))          

    #--- Show an orignial and reconstructed image 
    plt.figure()
    plt.title(dataset_name)
    if CHANNELS == 3:
        plt.imshow(A[0].squeeze().permute(1,2,0).cpu())
    else:
        plt.imshow(A[0].squeeze().cpu())
    plt.show()
    plt.figure()
    plt.title("Reconstruction")
    if CHANNELS == 3:
        plt.imshow(A_out[0].squeeze().permute(1,2,0).detach().cpu())
    else:
        plt.imshow(A_out[0].squeeze().detach().cpu())
    plt.show()
    #--- Show initial embedded space
    #Show_Partial_Embedding_Space(z_A, Y)


    """ --- Evaluation on place recognition --- """
    if Evaluate:
        print('Loading Evaluation data...')
    
        # --- Compute query embeddings ---
        #Eval_loader, _ = get_Tokyo_Query_dataloaders(batch_size=10, num_workers=WORKERS)        
        #Eval_loader_q, _ = get_Pittsburg30k_Eval_query_dataloaders(batch_size=10, num_workers=WORKERS)
        Eval_loader_q, _ = get_RobotCar_25_Eval_query_dataloaders(batch_size=10, num_workers=WORKERS)

        query_embs, y_q = [], []
    
        model.eval()
        with torch.no_grad():
            for q, y in Eval_loader_q:
                q = q.to(device)
                emb = compute_embeddings(model, q, device)   # returns tensor (B, D)
                query_embs.append(emb.cpu())                 # move to CPU
                y_q.append(y)
        
        with torch.no_grad():
            q_out, _, _ = model(q)
            q_out = torch.sigmoid(q_out) 
        #--- Show an orignial and reconstructed image 
        plt.figure()
        plt.title(dataset_name)
        if CHANNELS == 3:
            plt.imshow(q[0].squeeze().permute(1,2,0).cpu())
        else:
            plt.imshow(q[0].squeeze().cpu())
        plt.show()
        plt.figure()
        plt.title("Reconstruction")
        if CHANNELS == 3:
            plt.imshow(q_out[0].squeeze().permute(1,2,0).detach().cpu())
        else:
            plt.imshow(q_out[0].squeeze().detach().cpu())
        plt.show()
                
        query_embs = torch.cat(query_embs, dim=0).cpu()
        y_q = np.concatenate(y_q)
    
        # --- Compute database embeddings ---
        #Eval_loader, _ = get_Tokyo_Eval_dataloaders(batch_size=10, num_workers=WORKERS)
        #Eval_loader_db, db_dataset = get_Pittsburg30k_Eval_db_dataloaders(batch_size=10, num_workers=WORKERS)
        Eval_loader_db, db_dataset = get_RobotCar_25_Eval_db_dataloaders(batch_size=10, num_workers=WORKERS)
        db_embs, y_true = [], []
    
        with torch.no_grad():
            for data, y in Eval_loader_db:
                data = data.to(device)
                emb = compute_embeddings(model, data, device)
                db_embs.append(emb.cpu())
                y_true.append(y)
    
        db_embs = torch.cat(db_embs, dim=0).cpu()
        y_true = np.concatenate(y_true)
    
        # --- Evaluate ---
        print("Evaluating place recognition...")
        visualize_streaming_topk(model, Eval_loader_q, Eval_loader_db, db_dataset,
                                 device=device, metric="cosine", k=5, num_queries_to_show=5)
        
        recalls = Compute_recall_at_N(db_embs, query_embs, y_true, y_q, max_k=25)
    
        # Pick the k values we want to report
        k_vals = [1, 2, 3, 4, 5, 10, 15, 20, 25]
    
        # Plot only selected values
        plt.plot(k_vals, recalls[np.array(k_vals) - 1], marker="o")
        plt.xlabel("N (top-N)")
        plt.ylabel("Recall@N")
        plt.title("Recall Curve (sampled)")
        plt.grid(True)
        plt.show() 
        y_true =  torch.from_numpy(y_true)     
        #Show_Partial_Embedding_Space(db_embs, y_true)
        del query_embs, db_embs, y_true
        Eval_loader = None #--- Clear up memory

  
      
    """ --- Evaluation on "live" image stream --- """
    if Evaluate_on_Image_Stream:
        """---------------------------------------------------------------------------------------"""               
        """---    Run the Frozen model on a "live stream" of images and detect loop closure   ----""" 
        print("Running live stream processing")              
        #--- Thresholds 
        TAU_SAME = 0.93      #--- If last frame & current frame sim is greater than this, then in same place 
        TAU_CENTER = 0.5   #--- Current frame similiarty has to be greater than this to be in past place
        TAU_MEMBER = 0.52   #--- Min Sim the current frame needs to be to be a revisited place
        RATIO_MIN = 1.01    #--- How good the best match needs to be above the second best to be a revisited place
        TOP_M_CHECK = 50
        EXCLUDE_LAST = 10 
        FRAME_RATE = 20
           
        # exclude last N clusters from "revisit" eligibility 
        model.eval() 
        memory = OnlinePlaceMemory(dim=LATENT_DIM, tau_assign=TAU_CENTER, ema=0.1) 
        optimizer = torch.optim.Adam(model.ae.encoder.parameters(), lr=1e-5) 
        margin = 0.2 
        Kneg = 12 
        tau_new = 0.6 
        recent_cids = collections.deque(maxlen=EXCLUDE_LAST) 
        prev_emb = None 
           
        plt.figure("TAU Settings")
        plt.title("Tau Settings") 
        plt.text(0.1, 0.8, 'Tau Same: %.3f' %(TAU_SAME))
        plt.text(0.1, 0.7, 'Tau Center: %.3f' %(TAU_CENTER))
        plt.text(0.1, 0.6, 'Tau Member: %.3f' %(TAU_MEMBER))
        plt.text(0.1, 0.5, 'Min Ratio 1st, 2nd: %.3f' %(RATIO_MIN))
        plt.text(0.1, 0.4, 'Number Checked: %.1f' %(TOP_M_CHECK))
        plt.text(0.1, 0.3, 'Number last frames not to look at: %.1f' %(EXCLUDE_LAST))
        plt.text(0.1, 0.2, 'Triplet Margin: %.3f' %(margin))
        plt.text(0.1, 0.1, 'Frame Rate: %.1f' %(FRAME_RATE))
        plt.text(0.1, 0.0, dataset_name)
        plt.axis('off')
        
        Seq_eval_loader, dataset = get_RobotCar_Seq_Eval_db_dataloaders(batch_size=1, num_workers=WORKERS) 
        #Seq_eval_loader, dataset = get_RobotCar_12_Seq_Eval_dataloaders(batch_size=1, num_workers=WORKERS) 
        
        TP = TN = FP = FN = 0
        all_scores = []
        all_labels = []
        #--- Accounting for identifying a revisit and same place logic
        revisit_active = False          # Are we currently in a revisit episode?
        revisit_active_cid = None       # Which place (cid) is being revisited?
        # Optional: time-based latch (helps with brief drops)
        REVISIT_LATCH_FRAMES = int(1.5 * FRAME_RATE)  # e.g., 1.5 seconds
        latch_until = -1

        for i, (img, _, revisit) in enumerate(Seq_eval_loader):
            if i % FRAME_RATE != 0:
                continue
            sim_prev = 0.0           # so we can always use it in scoring
            best_member = None        # <-- reset each loop
            s_best = 0.0
            j_candidate = None        # track candidate cid for revisit UI text
           
            img = img.to(device)
            with torch.no_grad():
                _, z_aware, _ = model(img)
                z_t = z_aware.squeeze(0)
           
            state = "new_place"
            # Stage 0: same-as-previous
            if prev_emb is not None:
                sim_prev = float((z_t @ prev_emb).item()) #--- dot prod similarity 
                if sim_prev >= TAU_SAME:
                    state = "same_spot"
           
            # Stage 1+2: revisit
            if state != "same_spot":
                j, s_center = memory.nearest_center(z_t)
                if j >= 0 and (j not in recent_cids) and s_center >= TAU_CENTER:
                    members = [it for it in memory.items if it['cid'] == j]
                    if members:
                        members_slice = members[-TOP_M_CHECK:]
                        Zm = torch.stack([it['emb'] for it in members_slice])
                        sims = (Zm @ z_t)
                        s_sorted, order = torch.sort(sims, descending=True)
                        s_best = float(s_sorted[0])
                        s_second = float(s_sorted[1]) if len(s_sorted) > 1 else 0.0
                        ratio_ok = (s_second <= 1e-6) or (s_best / max(s_second, 1e-6) >= RATIO_MIN)
                        member_ok = s_best >= TAU_MEMBER
                        if member_ok and ratio_ok:
                            state = "revisit"
                            j_candidate = j
                            best_local = int(order[0].item())
                            best_member = members_slice[best_local]
          
            current_np = to_numpy_image(img[0])
            right_np, right_title = None, ""
           
            if state == "same_spot":
                left_title = f"t={i} SAME SPOT"
           
            elif state == "revisit":
                # Use safe fallbacks for UI text
                s_txt = f"{s_best:.2f}" if s_best is not None else "?"
                cid_txt = f"{j_candidate}" if j_candidate is not None else "?"
                left_title = f"t={i} REVISIT (cid={cid_txt}, best={s_txt})"
        
                if (best_member is not None) and (best_member.get('idx') is not None):
                    try:
                        right_np = get_image_by_index(dataset, best_member['idx'])
                        right_title = f"Best match (idx={best_member['idx']})"
                    except Exception:
                        right_np = None
            else:
                left_title = f"t={i} NEW PLACE"
          
            show_pair(current_np, right_np, left_title=left_title, right_title=right_title)
            prev_img_np = current_np
                               
            # --------- Triplet mining + (optional) fine-tune ---------- 
            mined = pick_triplet(z_t, i, memory, prev_z=prev_emb, Kneg=Kneg, tau_new=tau_new)
            if mined is not None: 
                z_pos, Zneg = mined 
                model.train() 
                _, z_aware_train, _ = model(img) 
                z_train = F.normalize(z_aware_train.squeeze(0), p=2, dim=0) 
                loss = triplet_loss(z_train, z_pos.to(device), Zneg.to(device), margin=margin) 
                optimizer.zero_grad() 
                loss.backward() 
                optimizer.step() 
                model.eval() 
                   
            # ---------- Update memory AFTER decision ---------- 
            # if state == "new_place":
            #     cid, is_new, s = memory.add(z_t, idx=i, force_new=True, update_center=False)
            # else:
            #     cid, is_new, s = memory.add(z_t, idx=i, force_new=False, update_center=True)

            if state == "new_place":
                cid, is_new, s = memory.add(z_t, idx=i, force_new=True, update_center=False)  # stores one new item
                last_cid = cid
            elif state == "revisit":
                cid = j_candidate if j_candidate is not None else last_cid
                if cid is not None:
                    memory.update_center(cid, z_t)  # EMA only, no append
                    last_cid = cid
            elif state == "same_spot":
                cid = last_cid
                if cid is not None:
                    memory.update_center(cid, z_t)  # or skip to freeze centers

         
            recent_cids.append(cid)
            prev_emb = z_t
            
            # --- Update (or clear) the revisit latch based on the decision/state ---
            if state == "revisit" and cid is not None:
                revisit_active = True
                revisit_active_cid = cid
                latch_until = i + REVISIT_LATCH_FRAMES
            
            elif state == "new_place":
                # We left the old place—end the revisit episode.
                revisit_active = False
                revisit_active_cid = None
                latch_until = -1
            
            elif state == "same_spot":
                # Stay latched only if we’re still in the same revisited place (cid may be last_cid)
                if not (revisit_active and (cid == revisit_active_cid)):
                    # We’re the same as previous frame, but not in the latched revisit place → do not count as positive
                    pass
            
            # Time-based un-latch safety (optional; shields against drift)
            if revisit_active and i > latch_until:
                revisit_active = False
                revisit_active_cid = None
            
            #--- Check for True positives, negatives, False positives, negatives
            # A frame is predicted positive if it's a fresh "revisit" OR it's "same_spot" while a revisit episode is active for the same cid.
            pred_positive = (state == "revisit") or (state == "same_spot" and revisit_active and (cid == revisit_active_cid))
            
            if revisit == 1 and pred_positive:
                TP += 1
            elif revisit == 1 and not pred_positive:
                FN += 1
            elif revisit == 0 and pred_positive:
                FP += 1
            else:
                TN += 1

          
            # Collect ground truth & score for plotting
            score = s_best if state == "revisit" else (sim_prev if prev_emb is not None else 0.0)
            all_scores.append(score)
            all_labels.append(int(revisit.item())) # ground truth 0/1

            mem_mb = process.memory_info().rss / (1024 * 1024)
            memory_usage.append(mem_mb)
            memory_len.append(len(memory.items))
              
        print("Total True positives: ", TP)
        print("Total True negatives: ", TN)
        print("Total False positives: ", FP)
        print("Total False negatives: ", FN)
        
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print(f"Recall: {recall:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"F1 Score: {f1:.3f}")
    
        plt.figure("Run")
        plt.title("Scoring") 
        plt.text(0.1, 0.8, 'True Positives: %.1f' %(TP))
        plt.text(0.1, 0.7, 'True Negatives: %.1f' %(TN))
        plt.text(0.1, 0.6, 'False Positives: %.1f' %(FP))
        plt.text(0.1, 0.5, 'False Negatives: %.1f' %(FN))
        plt.text(0.1, 0.4, 'Recall : %.3f' %(recall))
        plt.text(0.1, 0.3, 'Precision: %.3f' %(precision))
        plt.text(0.1, 0.2, 'F1 Score: %.3f' %(f1))
        plt.text(0.1, 0.0, dataset_name)
        plt.axis('off')   
    
        from sklearn.metrics import precision_recall_curve, average_precision_score
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Convert to numpy arrays
        y_true = np.array(all_labels)
        y_score = np.array(all_scores)

        
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        
        plt.figure(figsize=(6, 5))
        plt.plot(recall, precision, lw=2)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision–Recall Curve (AP = {ap:.3f})")
        plt.grid(True)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.show()  
        
        plt.figure("Memory Usage")
        plt.plot(memory_usage, label="Memory (MB)")
        plt.plot(memory_len, label="Place Memory Size", linestyle="--")
        plt.xlabel("Frame Index (sampled)")
        plt.ylabel("Memory Usage / Items")
        plt.title("Memory & Place Memory Growth")
        plt.legend()
        plt.grid(True)
        plt.show()
  