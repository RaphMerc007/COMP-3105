# TODO: Add necessary imports
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image


class_to_idx = {"Alarm_Clock": 0, "Bike": 1, "Desk_Lamp": 2, "Monitor": 3, "Mouse": 4, 
                "Oven": 5, "Pencil": 6, "Radio": 7, "Refrigerator": 8, "Screwdriver": 9}

def get_transforms():
  transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Make all images same size
    transforms.ToTensor(),  # Convert PIL to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
  ])
  return transform

def load_data(path, has_labels):
  images = []
  if has_labels:
    for folder in os.listdir(path): 
      if folder != '.DS_Store':
        folder_path = os.path.join(path, folder)
        for file in os.listdir(folder_path):
          if file.endswith('.jpg'):
            image = Image.open(os.path.join(folder_path, file)).convert('RGB')
            label = class_to_idx[folder] 
            image = get_transforms()(image)
            images.append((image, label))
  
  else:
    unlabelled_path = os.path.join(path, 'unlabelled')
    if os.path.exists(unlabelled_path):
      for file in os.listdir(unlabelled_path):
        if file.endswith('.jpg'):
          image = Image.open(os.path.join(unlabelled_path, file)).convert('RGB')
          image = get_transforms()(image)
          images.append(image)

  return images



def learn(path_to_in_domain, path_to_out_domain):
  in_train = load_data(path_to_in_domain, True)
  out_train = load_data(path_to_out_domain, False)
  print("data loaded")

  device = torch.device( "cuda" if torch.cuda.is_available() else "cpu")
  model = resnet18(weights=None) # not allowed to use pretrained model
  model.fc = nn.Linear(model.fc.in_features, 10)  # Change output to 10 classes
  model = model.to(device)
  
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)
  
  model.train()
  batch_size = 8
  
  try:
    # After some testing, does not get better after 18 epochs
    for epoch in range(40):
      # Shuffle data
      import random
      random.shuffle(in_train)
      correct = 0
      total = 0


      # Process in batches
      for i in range(0, len(in_train), batch_size):
        batch = in_train[i:i+batch_size]
        images = torch.stack([img for img, _ in batch]).to(device)
        labels = torch.tensor([label for _, label in batch]).to(device)

        total += labels.size(0)
        _, predicted = torch.max(model(images).data, 1) 
        correct += (predicted == labels).sum().item()

        # Train
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # if i % 30 == 0:
          # print(f"Epoch {epoch+1} {i/len(in_train):.2f} {correct/total:.4f}")



      print(f"Epoch {epoch+1} completed!!!!! {correct/total:.4f}")

  except KeyboardInterrupt:
    print(" Moving to test")

  model.class_to_idx = class_to_idx
  model.eval()
  return model







def compute_accuracy(path_to_eval_folder, model):
  eval_data = load_data(path_to_eval_folder, True)

  device = torch.device( "cuda" if torch.cuda.is_available() else "cpu")
  model = model.to(device)
  model.eval()  # Set to evaluation mode

  # Evaluate model
  correct = 0
  total = 0

  # Process in batches (more efficient)
  batch_size = 32
  with torch.no_grad():  # Don't compute gradients during evaluation
    for i in range(0, len(eval_data), batch_size):
      batch = eval_data[i:i+batch_size]
      images = torch.stack([img for img, _ in batch]).to(device)
      labels = torch.tensor([label for _, label in batch]).to(device)
      
      # Get predictions
      outputs = model(images)
      _, predicted = torch.max(outputs.data, 1)  # Get predicted class indices
      
      # Count correct predictions
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
  
  # Calculate accuracy
  accuracy = correct / total

  return accuracy







import time
time_start = time.time()



model = learn('./A4data/in-domain-train', './A4data/out-domain-train')

# Test on in-domain eval
in_acc = compute_accuracy('./A4data/in-domain-eval', model)
print(f"In-domain accuracy: {in_acc:.4f}")

# Test on out-domain eval
out_acc = compute_accuracy('./A4data/out-domain-eval', model)
print(f"Out-domain accuracy: {out_acc:.4f}")




time_end = time.time()
print(f"Time taken: {(time_end - time_start)/60:.2f} minutes")