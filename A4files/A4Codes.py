# TODO: Add necessary imports
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# import torchvision.transforms as transforms
import os
from PIL import Image
# from pathlib import Path


in_train = []
out_train = []
in_test = []
out_test = []
class_to_idx = {"Alarm_Clock": 0, "Bike": 1, "Desk_Lamp": 2, "Monitor": 3, "Mouse": 4, 
                "Oven": 5, "Pencil": 6, "Radio": 7, "Refrigerator": 8, "Screwdriver": 9}

def load_data(path_to_in_domain, path_to_out_domain):
  in_train.clear()
  out_train.clear()
  
  for folder in os.listdir(path_to_in_domain):
    if folder != '.DS_Store':
      folder_path = os.path.join(path_to_in_domain, folder)
      for file in os.listdir(folder_path):
        if file.endswith('.jpg'):
          image = Image.open(os.path.join(folder_path, file)).convert('RGB')
          label = class_to_idx[folder]  
          in_train.append((image, label))
  
  unlabelled_path = os.path.join(path_to_out_domain, 'unlabelled')
  if os.path.exists(unlabelled_path):
    for file in os.listdir(unlabelled_path):
      if file.endswith('.jpg'):
        image = Image.open(os.path.join(unlabelled_path, file)).convert('RGB')
        out_train.append(image)


def learn(path_to_in_domain, path_to_out_domain):
  # TODO: Implement training logic
  # - Load data
  # - Initialize model
  # - Train (consider using out-domain data to minimize out-domain accuracy)
  # - Return trained model
  pass


def compute_accuracy(path_to_eval_folder, model):

  # TODO: Implement evaluation logic
  # - Load evaluation data
  # - Run inference
  # - Calculate and return accuracy
  pass


load_data('./A4data/in-domain-train', './A4data/out-domain-train')