# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 10:52:31 2024

@author: djy41
"""
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
from config import EMA_TAU, EPS, MU_DIV, KAPPA_END, KAPPA_START, NUM_FEATURE_CLUSTERS, LAMBDA, BATCH_SIZE, LR, WORKERS, LATENT_DIM, NUM_CLASSES, dataset_name, CHANNELS, IHMC,EPOCHS
import torch
from tqdm import tqdm
from sklearn.manifold import TSNE

#************************************************************************
def plot_projection(x, colors, classes):
        
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    for i in range(classes):
        plt.scatter(x[colors == i, 0],
                    x[colors == i, 1])
  
    for i in range(classes):
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, i, fontsize=12)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
#***************************************************************************** 

#***************************************************************************** 
def Show_settings():
    plt.figure(1)
    plt.title("Run Settings") 
    plt.text(0.1, 0.8, 'Train Epochs: %.1f' %(EPOCHS))
    plt.text(0.1, 0.7, 'Mu Context embedding: %.1f' %(MU_DIV))
    plt.text(0.1, 0.6, 'Context embeddings: %.3f' %(NUM_FEATURE_CLUSTERS))
    plt.text(0.1, 0.5, 'Embeding Dimension: %.3f' %(LATENT_DIM))
    plt.text(0.1, 0.4, 'LAMBDA: %.3f' %(LAMBDA))
    plt.text(0.1, 0.3, 'Mu: %.3f' %(MU_DIV))
    plt.text(0.1, 0.2, 'Batch size: %.3f' %(BATCH_SIZE))
    plt.text(0.1, 0.1, dataset_name)
    plt.axis('off')

    if IHMC:
        plt.savefig("Settings.png")
    plt.show()
#***************************************************************************** 

#***************************************************************************** 
def Show_Results(Seed, Start_ACC, Start_NMI, Start_PUR, END_ACC,  END_NMI, END_PUR):
    plt.figure("Final Results")
    plt.title("Final Results") 
    plt.text(0.1, 0.9, 'Seed %.0f: ' %(Seed))
    plt.text(0.1, 0.8, 'Pretrain Acc: %.4f' %(Start_ACC))
    plt.text(0.1, 0.7, 'Pretrain NMI: %.4f' %(Start_NMI))
    plt.text(0.1, 0.6, 'Pretrain Purity: %.4f' %(Start_PUR))
    plt.text(0.1, 0.4, 'Ending Acc: %.4f' %(END_ACC))
    plt.text(0.1, 0.3, 'Ending NMI: %.4f' %(END_NMI))
    plt.text(0.1, 0.2, 'Ending Purity: %.4f' %(END_PUR))
    #plt.text(0.1, 0.1, dataset_name)
    plt.axis('off')

    if IHMC:
        plt.savefig("Final Results.png")
    plt.show()
#***************************************************************************** 

#*****************************************************************************
def Show_dataloader_data(img1, img2, labels):
   print("Batch shape: ", img1.shape)
   print(f"Feature batch shape: {img1.shape}")
   size = img1.shape
   size1 = size[2]
   size2 = size[-1]
   print(f"Image is {size1}x{size2}")
   print("Data Max value is: ",torch.max(img1))
   print("Data Min value is: ",torch.min(img1))
   
   if size[1] == 1:
       fig, ax = plt.subplots(1, 4, figsize=(10, 4))
       for i in range(4):
           ax[i].imshow(img1[i].squeeze()) # For 1 channel images
           ax[i].axis("off")
           ax[i].set_title(int(labels[i]))
           
       plt.show()
       
   if size[1] == 3:
       fig, ax = plt.subplots(1, 4, figsize=(10, 4))
       for i in range(4):
           ax[i].imshow(img1[i].squeeze().permute(1,2,0)) # For 3 channel images
           ax[i].axis("off")
           ax[i].set_title(int(labels[i]))
           
       plt.show()
#*****************************************************************************

#*****************************************************************************
def Show_Training_Loss(Loss_History):
        x = range(0,len(Loss_History))
        plt.figure()
        plt.title("Training loss") 
        plt.plot(x, Loss_History)    
        plt.xlabel(f"Training loss per Epoch-{EPOCHS}Epochs")     
        plt.axis('on')     

        if IHMC:
            plt.savefig("TrainingLoss.png")
            
        plt.show()
#*****************************************************************************

#*****************************************************************************
def Show_Variance(mse_error):
        x = range(0,len(mse_error))
        plt.figure('mse_error')
        plt.title("Variance") 
        plt.plot(x, mse_error)    
        plt.xlabel(f"Cluster #, - variance ={torch.var(mse_error)}")     
        plt.axis('on')     

        if IHMC:
            plt.savefig("TrainingLoss.png")
        plt.show
#*****************************************************************************

#*****************************************************************************
def Show_NMI_By_Epochs(NMI_History):
        x = range(0,len(NMI_History))
        plt.figure('NMI')
        plt.title("NMI per Update Epoch") 
        plt.plot(x, NMI_History)    
        plt.xlabel(f"NMI per Epoch-{FINE_TUNE_EPOCHS}Epochs")     
        plt.axis('on')     

        if IHMC:
            plt.savefig("NMI History.png")
        plt.show
#*****************************************************************************

#****** Plot the embeddings for one image ************************************
def Show_Component_Embeddings(z1, z2):
    z1 = z1.cpu()
    z2 = z2.cpu()
    
    plt.figure(figsize=(16,16))  
    plt.imshow(z1)
    plt.title("Common Embedding plot", fontsize=20)    
    plt.show() 
    plt.figure(figsize=(16,16))  
    plt.imshow(z2)
    plt.title("Peculiar Embedding plot")    

    if IHMC:
        plt.savefig("Common and Unique Embedding.png")
    plt.show() 
#*****************************************************************************    

#*****************************************************************************
def Show_Representation(decoded_common, label):
    fig, ax = plt.subplots(1, 10, figsize=(10, 4))
    if CHANNELS == 1:
        for i in range(10):
            ax[i].imshow(decoded_common[i].cpu().squeeze()) # For 1 channel images
            ax[i].axis("off")
            ax[i].set_title(int(label[i]))

        
    if CHANNELS == 3:
        for i in range(10):
            ax[i].imshow(decoded_common[i].cpu().squeeze().permute(1,2,0)) # For 3 channel images
            ax[i].axis("off")
            ax[i].set_title(int(label[i]))

    if IHMC:
        plt.savefig("Representations.png")
    plt.show() 
#***************************************************************************** 

    
#*****************************************************************************
def Show_Componet_Reconstructions(decoded_batch_common, decoded_batch_peculiar, labels):
    fig, ax = plt.subplots(1, 5, figsize=(10, 4))
    if CHANNELS == 1:
        for i in range(5):
            ax[i].imshow(decoded_batch_common[i].cpu().squeeze()) # For 1 channel images
            ax[i].axis("off")
            ax[i].set_title(int(labels[i]))
        plt.show()
        
        fig, ax = plt.subplots(1, 5, figsize=(10, 4))
        for i in range(5):
            ax[i].imshow(decoded_batch_peculiar[i].cpu().squeeze()) # For 1 channel images
            ax[i].axis("off")
            ax[i].set_title(int(labels[i]))
        plt.show()
        
    if CHANNELS == 3:
        for i in range(5):
            ax[i].imshow(decoded_batch_common[i].cpu().squeeze().permute(1,2,0)) # For 3 channel images
            ax[i].axis("off")
            ax[i].set_title(int(labels[i]))
        plt.show()
        
        fig, ax = plt.subplots(1, 5, figsize=(10, 4))
        for i in range(5):
            ax[i].imshow(decoded_batch_peculiar[i].cpu().squeeze().permute(1,2,0)) # For 3 channel images
            ax[i].axis("off")
            ax[i].set_title(int(labels[i]))
        plt.show()
    if IHMC:
        plt.savefig("Reconstructions.png")
#*****************************************************************************    

#*****************************************************************************
def Show_Complete_Reconstructions(x, x_hat):
    fig, axes = plt.subplots(2, 2)
    for i in range(2):
        if CHANNELS == 3:
            axes[0, i].imshow(x[i].cpu().squeeze().permute(1, 2, 0))
            axes[1, i].imshow(x_hat[i].detach().cpu().squeeze().permute(1, 2, 0))
        else:
            axes[0, i].imshow(x[i].cpu().squeeze())
            axes[1, i].imshow(x_hat[i].detach().cpu().squeeze())
        axes[0, i].axis('off')
        axes[1, i].axis('off')

    if IHMC:
        plt.savefig("Complete Reconstruction.png")
    plt.show()
#*****************************************************************************  
    
#******* Show the embedding space using true labels and t-SNE ****************
def Show_Embedding_Space(z1, z2, labels):
    z1 = z1.detach().cpu()
    z2 = z2.detach().cpu()
    latent1 = torch.flatten(z1,start_dim=1)
    latent2 = torch.flatten(z2,start_dim=1)
    latent1 = latent1.cpu()
    latent2 = latent2.cpu()
    x1 = latent1.numpy()
    x2 = latent2.numpy()
    #labels = labels.cpu()
    #labels = labels.numpy()

    tsne = TSNE(n_components=2, verbose=0, perplexity=30, max_iter=500)
    X_tsne = tsne.fit_transform(x1)       
    plot_projection(X_tsne, labels, NUM_CLASSES)
    plt.title("Common Embedding Space", fontsize=28)  

    if IHMC:
        plt.savefig("Common Embedded Space.png")
    plt.show()
        
    tsne = TSNE(n_components=2, verbose=0, perplexity=30, max_iter=500)
    X_tsne = tsne.fit_transform(x2)       
    plot_projection(X_tsne, labels, NUM_CLASSES)
    plt.title("Peculiar Embedding Space", fontsize=28)  
 
    if IHMC:
        plt.savefig("Unique Embedded Space.png")
    plt.show()
#*****************************************************************************    

def tensor_to_numpy(t):
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    elif isinstance(t, (list, tuple)):
        return type(t)(tensor_to_numpy(x) for x in t)
    elif isinstance(t, dict):
        return {k: tensor_to_numpy(v) for k, v in t.items()}
    else:
        return t  # leave non-tensor objects untouched
    
#******* Show the embedding space of one latent using true labels and t-SNE ****************
def Show_Partial_Embedding_Space(z, y_true):       
    z = z.detach().cpu()  
    
    # Remap labels to 0..num_classes-1
    unique_labels, y_new = torch.unique(y_true, return_inverse=True)
    num_classes = len(unique_labels)
    
    latent1 = torch.flatten(z, start_dim=1)  
    x1 = tensor_to_numpy(latent1)
    x1 = x1 + 1e-8 * np.random.randn(*x1.shape)
    
    tsne = TSNE(n_components=2, verbose=0, perplexity=20, max_iter=500)
    X_tsne = tsne.fit_transform(x1)       

    # Pass remapped labels to projection function
    plot_projection(X_tsne, y_new, num_classes)
    plt.title("Embedding Space")  

    if IHMC:
        plt.savefig("Partial Embeded Space.png")
    plt.show()
#*****************************************************************************        
    
    