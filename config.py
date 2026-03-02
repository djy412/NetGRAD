# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 10:45:46 2025
@author: djy41
"""
BATCH_SIZE = 50
WORKERS = 8
LATENT_DIM = 512
# Pittsburg30k  417; RobotCar 642: Tokyo 24/7 1750
NUM_CLASSES = 642
NUM_FEATURE_CLUSTERS = 64
NUM_VIEWS = 2

CHANNELS = 1
IHMC = False  

EPOCHS = 8
LR = 1e-3
LAMBDA = 1
BETA = 1
MARGIN = 0.5

# Regularizer + assignment sharpness
KAPPA_START = 5.0      # soft assignments early
KAPPA_END   = 15.0     # moderately sharp later
MU_DIV      = 0.1      # single regularizer weight
EPS         = 1e-8

EMA_TAU     = 0.0      # keep OFF to stick with exactly one regularizer (set >0 to enable EMA nudge)

UPDATE_INTERVAL = 10
TOLERANCE = 0.001 #--- How close to the last estimage is good enough
PATIENCE = 3

#dataset_name = 'MULTI_STL-10'
#dataset_name = 'Tokyo 24/7'
#dataset_name = 'Pittsburg30k'
dataset_name = 'RobotCar'

