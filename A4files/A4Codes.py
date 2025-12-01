import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image


# Wrapper class to hold the model
class ModelWrapper:
  def __init__(self, model, class_to_idx, num_classes):
    self.model = model
    self.class_to_idx = class_to_idx
    self.num_classes = num_classes

DATASET_MEAN = []
DATASET_STD = []

def get_transforms(augment, normalize=False):
  # Dataset-specific normalization constants (computed from training data)
  global DATASET_MEAN, DATASET_STD
  if augment:
    # Training transforms with aggressive data augmentation
    transform_list = [
      transforms.Resize((280, 280)),  # larger for cropping
      transforms.RandomCrop((224, 224)),  # Random crop for augmentation
      transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip
      transforms.RandomRotation(degrees=15),  # Random rotation up to 15 degrees
      transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.4, hue=0.3),  # color jitter
      transforms.ToTensor()
    ]
    if normalize:
      transform_list.append(transforms.Normalize(mean=DATASET_MEAN, std=DATASET_STD))
    transform = transforms.Compose(transform_list)
  else:
    # Evaluation transforms (no augmentation)
    transform_list = [
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
    ]
    if normalize:
      transform_list.append(transforms.Normalize(mean=DATASET_MEAN, std=DATASET_STD))
    transform = transforms.Compose(transform_list)
  return transform


# Discover class names from folder structure and create class_to_idx mapping
def discover_classes(path):
  classes = []
  # list all folders in the path
  for folder in os.listdir(path):
    if folder != '.DS_Store' and os.path.isdir(os.path.join(path, folder)):
      classes.append(folder)
  classes.sort()  # Sort for consistent ordering
  class_to_idx = {}
  for idx, cls in enumerate(classes):
      class_to_idx[cls] = idx
  return class_to_idx, len(classes)


# Load data from path
def load_data(path, has_labels, class_to_idx=None):
  images = []

  # out domain train does not have labels
  if has_labels:
    if class_to_idx is None:
      class_to_idx, _ = discover_classes(path)

    for folder in os.listdir(path): 
      if folder != '.DS_Store' and folder in class_to_idx:
        folder_path = os.path.join(path, folder)
        for file in os.listdir(folder_path):
          if file.endswith('.jpg'):
            image = Image.open(os.path.join(folder_path, file)).convert('RGB')
            label = class_to_idx[folder] 
            # append the image and label to the list
            images.append((image, label))
  
  else:
    unlabelled_path = os.path.join(path, 'unlabelled')
    if os.path.exists(unlabelled_path):
      for file in os.listdir(unlabelled_path):
        if file.endswith('.jpg'):
          image = Image.open(os.path.join(unlabelled_path, file)).convert('RGB')
          # append the image to the list
          images.append(image)

  return images


# Compute dataset mean and std for normalization
def compute_dataset_stats(in_train, out_train, class_to_idx):  
  # Load data
  # Transform to tensor (resize and convert to tensor only, no normalization)
  transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
  ])
  
  # Collect all images
  all_images = []
  for img, _ in in_train:
    all_images.append(transform(img))
  for img in out_train:
    all_images.append(transform(img))
  
  # Stack into tensor
  images_tensor = torch.stack(all_images)
  
  # Compute mean and std across all images and all pixels
  # Shape: [num_images, channels, height, width]
  mean = images_tensor.mean(dim=(0, 2, 3))  # Mean across batch, height, width
  std = images_tensor.std(dim=(0, 2, 3))    # Std across batch, height, width
    
  return mean.tolist(), std.tolist()


# Compute entropy of predictions (for entropy maximization)
def compute_entropy(outputs):
  probs = torch.nn.functional.softmax(outputs, dim=1)
  entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
  return entropy.mean()


# train the model using entropy-based approach
def learn(path_to_in_domain, path_to_out_domain):  
  # Discover classes dynamically from in-domain training data
  class_to_idx, num_classes = discover_classes(path_to_in_domain)
  
  in_train = load_data(path_to_in_domain, has_labels=True, class_to_idx=class_to_idx)
  out_train = load_data(path_to_out_domain, has_labels=False)
  mean, std = compute_dataset_stats(in_train, out_train, class_to_idx)
  global DATASET_MEAN, DATASET_STD

  DATASET_MEAN = mean
  DATASET_STD = std
  # Use normalization consistently for all training data
  augment_transform = get_transforms(augment=True, normalize=True)
  no_augment_transform = get_transforms(augment=False, normalize=True)
  
  # Start with non-augmented versions
  in_train_augmented = [(no_augment_transform(img), label) for img, label in in_train]
  out_train_augmented = [no_augment_transform(img) for img in out_train]
        
  # Augment data
  for i in range(6):
    for img, label in in_train:
      in_train_augmented.append((augment_transform(img), label))

      # Balance out-domain data to match in-domain size
    while len(out_train_augmented) != len(in_train_augmented):
      if len(out_train_augmented) < len(in_train_augmented):
        for img in out_train:
          out_train_augmented.append(augment_transform(img))
          if len(out_train_augmented) >= len(in_train_augmented):
            break
      if len(out_train_augmented) > len(in_train_augmented):
        for img, label in in_train:
          in_train_augmented.append((augment_transform(img), label))
          if len(out_train_augmented) <= len(in_train_augmented):
            break
    

  # Create single model
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = resnet18(weights=None)
  model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, num_classes)
  )
  model = model.to(device)

  # Loss functions
  criterion_in = nn.CrossEntropyLoss()  # For in-domain: minimize cross-entropy
  optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=0.0001)
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.8)

  model.train()
  batch_size = 64
  patience = 6
  min_improvement = 0.01
  entropy_weight = 0.2  # Weight for entropy loss (can tune this)

  best_acc = 0.0
  no_improve = 0
  
  for epoch in range(100):
    import random
    random.shuffle(in_train_augmented)
    random.shuffle(out_train_augmented)
    
    correct = 0
    total = 0
    total_loss = 0.0

    epoch_ce_loss = 0.0
    epoch_entropy_loss = 0.0
    epoch_out_conf = 0.0

    # Process in batches - alternate between in-domain and out-domain
    num_batches = max(len(in_train_augmented), len(out_train_augmented)) // batch_size
    
    for batch_idx in range(num_batches):
      # Get in-domain batch
      in_start = (batch_idx * batch_size) % len(in_train_augmented)
      in_batch = in_train_augmented[in_start:in_start + batch_size]
      
      in_images = []
      in_labels = []
      for img, label in in_batch:
        in_images.append(img)
        in_labels.append(label)
      
      in_images_tensor = torch.stack(in_images).to(device)
      in_labels_tensor = torch.tensor(in_labels).to(device)

      # Get out-domain batch
      out_start = (batch_idx * batch_size) % len(out_train_augmented)
      out_batch = out_train_augmented[out_start:out_start + batch_size]
      
      out_images = []
      for img in out_batch:
        out_images.append(img)
      
      out_images_tensor = torch.stack(out_images).to(device)

      # Forward pass and compute losses
      optimizer.zero_grad()
      
      # In-domain: minimize cross-entropy (standard classification loss)
      in_outputs = model(in_images_tensor)
      in_loss = criterion_in(in_outputs, in_labels_tensor)
      
      # Out-domain: maximize entropy (minimize negative entropy)
      out_outputs = model(out_images_tensor)
      out_entropy = compute_entropy(out_outputs)

      batch_ce_loss = in_loss.item()
      batch_entropy_loss = out_entropy.item()

      # Compute out-domain confidence (softmax max prob)
      with torch.no_grad():
          out_probs = torch.softmax(out_outputs, dim=1)
          batch_max_conf = out_probs.max(dim=1).values.mean().item()

      epoch_ce_loss += batch_ce_loss
      epoch_entropy_loss += batch_entropy_loss
      epoch_out_conf += batch_max_conf

      out_loss = -out_entropy  # Negative because we want to maximize entropy
      
      # Combined loss
      loss = in_loss + entropy_weight * out_loss
      loss.backward()
      optimizer.step()

      # Track accuracy (only on in-domain)
      total += in_labels_tensor.size(0)
      _, predicted = torch.max(in_outputs.data, 1)
      correct += (predicted == in_labels_tensor).sum().item()
      total_loss += loss.item()

    scheduler.step()
    current_acc = correct / total

    # Convergence check
    improvement = current_acc - best_acc
    if improvement > min_improvement:
      best_acc = current_acc
      no_improve = 0
    else:
      no_improve += 1
        
    if no_improve >= patience and current_acc > 0.98 or no_improve >= patience * 2:
      break

  
  model.eval()

  return ModelWrapper(model, class_to_idx, num_classes)


def compute_accuracy(path_to_eval_folder, model):
  # Extract model and class info from wrapper
  model_eval = model.model
  class_to_idx = model.class_to_idx
  num_classes = model.num_classes
  
  eval_data_pil = load_data(path_to_eval_folder, True, class_to_idx)
  # Transform PIL images to tensors (use same normalization as training)
  eval_transform = get_transforms(False, normalize=True)
  eval_data = []
  for img, label in eval_data_pil:
      transformed_img = eval_transform(img)
      eval_data.append((transformed_img, label))

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model_eval = model_eval.to(device)
  model_eval.eval()

  # Evaluate model
  correct = 0
  total = 0

  # Process in batches
  batch_size = 64
  with torch.no_grad():
    for i in range(0, len(eval_data), batch_size):
      batch = eval_data[i:i+batch_size]
      image_list = []
      label_list = []
      for img, label in batch:
          image_list.append(img)
          label_list.append(label)
      images = torch.stack(image_list).to(device)
      labels = torch.tensor(label_list).to(device)
      
      # Get predictions
      outputs = model_eval(images)
      _, predicted = torch.max(outputs.data, 1)

      correct += (predicted == labels).sum().item()
      total += len(images)

  accuracy = correct / total
  return accuracy