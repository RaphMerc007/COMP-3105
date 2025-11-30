# TODO: Add necessary imports
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


def get_transforms(augment, normalize=False):
  if augment:
    # Training transforms with aggressive data augmentation
    transform = transforms.Compose([
      transforms.Resize((280, 280)),  # larger for cropping
      transforms.RandomCrop((224, 224)),  # Random crop for augmentation
      transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip
      transforms.RandomRotation(degrees=15),  # Random rotation up to 15 degrees
      transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.4, hue=0.3),  # color jitter
      transforms.ToTensor()
    ])
  else:
    # Evaluation transforms (no augmentation)
    transform = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
    ])
    if normalize:
      transform.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
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
  print(f"discovered classes: {classes}")
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

  augment_transform = get_transforms(augment=True)
  no_augment_transform = get_transforms(augment=False)
  
  print("Creating augmented dataset...")
  # Start with non-augmented versions
  in_train_augmented = [(no_augment_transform(img), label) for img, label in in_train]
  out_train_augmented = [no_augment_transform(img) for img in out_train]
  
  # Augment data
  for i in range(6):
    if i == 5:
      augment_transform = get_transforms(True, True)
      
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
  
  print(f"Augmented dataset created: {len(in_train_augmented)} in-domain images, {len(out_train_augmented)} out-domain images")

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
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.6)

  model.train()
  batch_size = 64
  patience = 20
  min_improvement = 0.002
  entropy_weight = 0.2  # Weight for entropy loss (can tune this)

  try:
    best_acc = 0.0
    no_improve = 0
    
    for epoch in range(100):
      import random
      random.shuffle(in_train_augmented)
      random.shuffle(out_train_augmented)
      
      correct = 0
      total = 0
      total_loss = 0.0

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
      avg_loss = total_loss / num_batches
      
      # Convergence check
      improvement = current_acc - best_acc
      if improvement > min_improvement:
        best_acc = current_acc
        no_improve = 0
        print(f"Epoch {epoch+1} → New best accuracy: {best_acc:.4f} (+{improvement:.4f}), Loss: {avg_loss:.4f}")
      else:
        no_improve += 1
        print(f"Epoch {epoch+1} → No improvement for {no_improve} epochs (best: {best_acc:.4f}, current: {current_acc:.4f}, Loss: {avg_loss:.4f})")
      
      if current_acc > 0.99:
        print(f"\nEpoch {epoch+1} → High accuracy reached ({current_acc:.4f}), stopping to prevent overfitting.")
        break
          
      if no_improve >= patience:
        print(f"\nEpoch {epoch+1} → Convergence reached! No improvement for {patience} epochs. Stopping training.")
        print(f"Best accuracy: {best_acc:.4f}")
        break

  except KeyboardInterrupt:
    print("Training interrupted")

  model.eval()
  return ModelWrapper(model, class_to_idx, num_classes)







def compute_accuracy(path_to_eval_folder, model):
  # Extract model and class info from wrapper
  model_eval = model.model
  class_to_idx = model.class_to_idx
  num_classes = model.num_classes
  
  eval_data_pil = load_data(path_to_eval_folder, True, class_to_idx)
  # Transform PIL images to tensors
  eval_transform = get_transforms(False)
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
  batch_size = 32
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