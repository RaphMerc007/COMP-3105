# TODO: Add necessary imports
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image


class_to_idx = {"Alarm_Clock": 0, "Bike": 1, "Desk_Lamp": 2, "Monitor": 3, "Mouse": 4, 
                "Oven": 5, "Pencil": 6, "Radio": 7, "Refrigerator": 8, "Screwdriver": 9}

def get_transforms(augment):
  if augment:
    # Training transforms with data augmentation
    transform = transforms.Compose([
      transforms.Resize((256, 256)),  # Slightly larger for cropping
      transforms.RandomCrop((224, 224)),  # Random crop for augmentation
      transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip
      transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Color jitter
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
  else:
    # Evaluation transforms (no augmentation)
    transform = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
            images.append((image, label))
  
  else:
    unlabelled_path = os.path.join(path, 'unlabelled')
    if os.path.exists(unlabelled_path):
      for file in os.listdir(unlabelled_path):
        if file.endswith('.jpg'):
          image = Image.open(os.path.join(unlabelled_path, file)).convert('RGB')
          images.append(image)

  return images



def learn(path_to_in_domain, path_to_out_domain):
  in_train = load_data(path_to_in_domain, True)
  out_train = load_data(path_to_out_domain, False)

  # Create augmented versions (9 additional versions to make ~10x larger)
  augment_transform = get_transforms(True)
  no_augment_transform = get_transforms(False)
  
  print("Creating augmented dataset...")
  # Start with non-augmented versions
  in_train_augmented = [(no_augment_transform(img), label) for img, label in in_train]
  out_train_augmented = [no_augment_transform(img) for img in out_train]
  
  # Add 9 augmented versions of each image
  for _ in range(4):
    for img, label in in_train:
      in_train_augmented.append((augment_transform(img), label))
   
    while len(out_train_augmented) < len(in_train_augmented):
      for img in out_train:
        out_train_augmented.append(augment_transform(img))
        if len(out_train_augmented) >= len(in_train_augmented):
          break
  
  
  print(f"Augmented dataset created: {len(in_train_augmented)} in-domain images, {len(out_train_augmented)} out-domain images")

  categorise_data_train = []
  for (image, _) in in_train_augmented:
    categorise_data_train.append((image, 0))
  
  for image in out_train_augmented:
    categorise_data_train.append((image, 1))

  print("data loaded")

  device_categorise = torch.device( "cuda" if torch.cuda.is_available() else "cpu")
  model_categorise = resnet18(weights=None) # not allowed to use pretrained model
  model_categorise.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model_categorise.fc.in_features, 2)
  )
  model_categorise = model_categorise.to(device_categorise)
  
  criterion_categorise = nn.CrossEntropyLoss()
  optimizer_categorise = optim.Adam(model_categorise.parameters(), lr=0.002, weight_decay=0.0001)
  
  model_categorise.train()
  batch_size = 16
  
  try:
    for epoch in range(80):
      # Shuffle data
      import random
      random.shuffle(categorise_data_train)
      correct = 0
      total = 0


      # Process in batches
      for i in range(0, len(categorise_data_train), batch_size):
        batch = categorise_data_train[i:i+batch_size]
        images = torch.stack([img for img, _ in batch]).to(device_categorise)
        labels = torch.tensor([label for _, label in batch]).to(device_categorise)

        # Train
        optimizer_categorise.zero_grad()
        outputs = model_categorise(images)
        loss = criterion_categorise(outputs, labels)
        loss.backward()
        optimizer_categorise.step()


        total += labels.size(0)
        _, predicted = torch.max(outputs.data, 1) 
        correct += (predicted == labels).sum().item()

      print(f"Epoch {epoch+1} completed!!!!! {correct/total:.4f}")


  except KeyboardInterrupt:
    print(" Categorisation model trained")

  print(" Moving to classify")








  device_classify = torch.device( "cuda" if torch.cuda.is_available() else "cpu")
  model_classify = resnet18(weights=None) # not allowed to use pretrained model
  model_classify.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model_classify.fc.in_features, 10)
  )
  model_classify = model_classify.to(device_classify)

  criterion_classify = nn.CrossEntropyLoss()
  optimizer_classify = optim.Adam(model_classify.parameters(), lr=0.001, weight_decay=0.001)

  # Use the pre-augmented in-domain data
  model_classify.train()
  batch_size = 16

  try:
    for epoch in range(60):
      # Shuffle data
      import random
      random.shuffle(in_train_augmented)
      correct = 0
      total = 0


      # Process in batches
      for i in range(0, len(in_train_augmented), batch_size):
        batch = in_train_augmented[i:i+batch_size]
        images = torch.stack([img for img, _ in batch]).to(device_classify)
        labels = torch.tensor([label for _, label in batch]).to(device_classify)

        # Train
        optimizer_classify.zero_grad()
        outputs = model_classify(images)
        loss = criterion_classify(outputs, labels)
        loss.backward()
        optimizer_classify.step()

        # Calculate accuracy AFTER training
        total += labels.size(0)
        _, predicted = torch.max(outputs.data, 1) 
        correct += (predicted == labels).sum().item()



      print(f"Epoch {epoch+1} completed!!!!! {correct/total:.4f}")

  except KeyboardInterrupt:
    print(" Moving to test")

  model_classify.class_to_idx = class_to_idx
  model_classify.eval()
  return model_classify, model_categorise







def compute_accuracy(path_to_eval_folder, model_classify, model_categorise):
  eval_data_pil = load_data(path_to_eval_folder, True)
  # Transform PIL images to tensors
  eval_transform = get_transforms(False)
  eval_data = [(eval_transform(img), label) for img, label in eval_data_pil]

  device_classify = torch.device( "cuda" if torch.cuda.is_available() else "cpu")
  device_categorise = torch.device( "cuda" if torch.cuda.is_available() else "cpu")
  model_classify = model_classify.to(device_classify)
  model_categorise = model_categorise.to(device_categorise)
  model_classify.eval()  # Set to evaluation mode
  model_categorise.eval()  # Set to evaluation mode

  # Determine true category label (0 for in-domain, 1 for out-domain)
  true_category = 0 if "in-domain" in path_to_eval_folder else 1

  # Evaluate model
  correct = 0
  total = 0
  categorise_correct = 0
  categorise_total = 0

  # Process in batches (more efficient)
  batch_size = 32
  with torch.no_grad():  # Don't compute gradients during evaluation
    for i in range(0, len(eval_data), batch_size):
      batch = eval_data[i:i+batch_size]
      images = torch.stack([img for img, _ in batch]).to(device_categorise)
      labels = torch.tensor([label for _, label in batch]).to(device_classify)
      
      # First, determine if each image is in-domain or out-domain
      categorise_outputs = model_categorise(images)
      _, predicted_categorise = torch.max(categorise_outputs.data, 1)
      
      # Track categorization accuracy
      categorise_total += len(images)
      categorise_correct += (predicted_categorise == true_category).sum().item()
      
      # Separate images and labels based on in/out domain prediction
      images_to_classify = []
      labels_to_classify = []
      indices_to_guess = []
      
      for idx in range(len(images)):
        if predicted_categorise[idx] == 0:  # In-domain
          images_to_classify.append(images[idx])
          labels_to_classify.append(labels[idx])
        else:  # Out-domain
          indices_to_guess.append(idx)
      
      # For in-domain images: classify them
      if len(images_to_classify) > 0:
        images_to_classify_tensor = torch.stack(images_to_classify).to(device_classify)
        labels_to_classify_tensor = torch.tensor(labels_to_classify).to(device_classify)
        outputs = model_classify(images_to_classify_tensor)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels_to_classify_tensor).sum().item()
        total += len(images_to_classify)
      
      # For out-domain images: just guess randomly
      for idx in indices_to_guess:
        guess = random.randint(0, 9)
        if guess == labels[idx].item():
          correct += 1
        total += 1
  
  # Calculate accuracies
  accuracy = correct / total
  categorise_accuracy = categorise_correct / categorise_total
  
  print(f"Categorization accuracy: {categorise_accuracy:.4f}")

  return accuracy







import time
time_start = time.time()



model_classify, model_categorise = learn('./A4data/in-domain-train', './A4data/out-domain-train')

# Test on in-domain eval
in_acc = compute_accuracy('./A4data/in-domain-eval', model_classify, model_categorise)
print(f"In-domain accuracy: {in_acc:.4f}")

# Test on out-domain eval
out_acc = compute_accuracy('./A4data/out-domain-eval', model_classify, model_categorise)
print(f"Out-domain accuracy: {out_acc:.4f}")




time_end = time.time()
print(f"Time taken: {(time_end - time_start)/60:.2f} minutes")