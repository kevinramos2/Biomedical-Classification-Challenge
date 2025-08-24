# Funciones auxiliares
import numpy as np  
import random
import torch

def semilla(seed=42):
  # Se fija la semilla para reproducibilidad en numpy, random y torch
  
  random.seed(seed)
  np.random.seed(seed)
  if torch.cuda.is_available():
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)