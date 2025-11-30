# TODO: Add necessary imports
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.models import resnet18
from PIL import Image



# Wrapper class to hold both categorization and classification models
class ModelWrapper:
  def __init__(self, model_classify, model_categorise, class_to_idx, num_classes):
    self.model_classify = model_classify
    self.model_categorise = model_categorise
    self.class_to_idx = class_to_idx
    self.num_classes = num_classes


def get_transforms(augment):
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



# train the model
def learn(path_to_in_domain, path_to_out_domain):
  # Discover classes dynamically from in-domain training data
  class_to_idx, num_classes = discover_classes(path_to_in_domain)
  
  in_train = load_data(path_to_in_domain, has_labels=True, class_to_idx=class_to_idx)
  out_train = load_data(path_to_out_domain, has_labels=False)

  augment_transform = get_transforms(augment=True)
  no_augment_transform = get_transforms(augment=False)
  
  print("Creating augmented dataset...")
  # Start with non-augmented versions
  in_train_augmented = []
  for img, label in in_train:
    transformed_img = no_augment_transform(img)
    in_train_augmented.append((transformed_img, label))
  
  out_train_augmented = []
  for img in out_train:
    transformed_img = no_augment_transform(img)
    out_train_augmented.append(transformed_img)
  
  for _ in range(3):
    # augment the in-domain data
    for img, label in in_train:
      in_train_augmented.append((augment_transform(img), label))

    # make sure that the out-domain data is the same size as the in-domain data
    while len(out_train_augmented) != len(in_train_augmented):

      if len(out_train_augmented) < len(in_train_augmented):
        # add more out-domain data
        for img in out_train:
          out_train_augmented.append(augment_transform(img))
          if len(out_train_augmented) >= len(in_train_augmented):
            break
          
      if len(out_train_augmented) > len(in_train_augmented):
        # remove some in-domain data
        for img, label in in_train:
          in_train_augmented.append((augment_transform(img), label))
          if len(out_train_augmented) <= len(in_train_augmented):
            break

  print(f"Augmented dataset created: {len(in_train_augmented)} in-domain images, {len(out_train_augmented)} out-domain images")



  #============================= Classification Model =======================================

  # create the classification model
  device_classify = torch.device( "cuda" if torch.cuda.is_available() else "cpu")
  model_classify = resnet18(weights=None) # not allowed to use pretrained model
  model_classify.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model_classify.fc.in_features, num_classes)
  )
  model_classify = model_classify.to(device_classify)

  criterion_classify = nn.CrossEntropyLoss()
  optimizer_classify = optim.Adam(model_classify.parameters(), lr=0.002, weight_decay=0.0001)
  scheduler_classify = optim.lr_scheduler.StepLR(optimizer_classify, step_size=8, gamma=0.6)

  # Use the pre-augmented in-domain data
  model_classify.train()
  # Convergence check parameters
  batch_size = 64
  patience_classify = 20  # amount of epochs to wait for improvement
  min_improvement_classify = 0.002  # Only count improvements > 0.2% as meaningful

  try:
    for n in range(5):
      best_acc_classify = 0.0 # best accuracy so far
      no_improve_classify = 0 # amount of epochs without improvement

      # in domain training loop
      for epoch in range(100):
        # Shuffle data
        import random
        random.shuffle(in_train_augmented)
        correct = 0
        total = 0
        ce_loss_epoch = 0
        batch_count = 0
        entropy_epoch = 0
        conf_epoch = 0

        # Process in batches
        for i in range(0, len(in_train_augmented), batch_size):
          batch = in_train_augmented[i:i+batch_size]

          image_list = []
          label_list = []
          for img, label in batch:
            image_list.append(img)
            label_list.append(label)

          images = torch.stack(image_list).to(device_classify)
          labels = torch.tensor(label_list).to(device_classify)

          # Train
          optimizer_classify.zero_grad()
          outputs = model_classify(images)
          loss = criterion_classify(outputs, labels)
          loss.backward()
          optimizer_classify.step()

          # Accumulate graph metrics
          ce_loss_epoch += loss.item()
          batch_count += 1

          # Calculate accuracy training
          total += labels.size(0)
          _, predicted = torch.max(outputs.data, 1) # get the predicted class
          correct += (predicted == labels).sum().item()

        scheduler_classify.step()
        current_acc = correct / total

        # Avg metrics
        train_acc_history.append(current_acc)

        avg_ce = ce_loss_epoch / batch_count
        in_ce_loss_history.append(avg_ce)

        with torch.no_grad():
          ood_images = torch.stack(out_train_augmented[:256]).to(device_classify)

          ood_logits = model_classify(ood_images)
          probs = torch.softmax(ood_logits, dim=1)

          # entropy
          entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
          out_entropy_history.append(entropy.mean().item())

          # confidence
          conf = probs.max(dim=1)[0]
          out_conf_history.append(conf.mean().item())

        # Convergence check with meaningful improvement threshold
        improvement = current_acc - best_acc_classify
        if improvement > min_improvement_classify:
          best_acc_classify = current_acc
          no_improve_classify = 0
          print(f"epoch {epoch+1} → New best accuracy: {best_acc_classify:.4f} (+{improvement:.4f})")
        else:
          no_improve_classify += 1
          print(f"epoch {epoch+1} → No improvement for {no_improve_classify} epochs (best: {best_acc_classify:.4f}, current: {current_acc:.4f})")
        
        # Early stop if accuracy is very high (likely overfitting)
        if current_acc > 0.99:
          print(f"\nepoch {epoch+1} → High accuracy reached ({current_acc:.4f}), stopping to prevent overfitting.")
          break
          
        if no_improve_classify >= patience_classify:
          print(f"\nepoch {epoch+1} → Convergence reached! No improvement for {patience_classify} epochs. Stopping training.")
          print(f"epoch {epoch+1} → Best accuracy: {best_acc_classify:.4f}")
          break

      if n == 4:
        continue

      # Test on in-domain eval
      in_acc = compute_accuracy('./A4data/in-domain-eval', ModelWrapper(model_classify, None, class_to_idx, num_classes))
      print(f"In-domain accuracy: {in_acc:.4f}")

      # Test on out-domain eval
      out_acc = compute_accuracy('./A4data/out-domain-eval', ModelWrapper(model_classify, None, class_to_idx, num_classes))
      print(f"Out-domain accuracy: {out_acc:.4f}")

      time_end = time.time()
      print(f"Time taken: {(time_end - time_start)/60:.2f} minutes")
      model_classify.train()

      # Adversarial training on out-domain: train model to predict least likely class
      for epoch in range(20):
        # Shuffle data
        import random
        random.shuffle(out_train_augmented)
        correct = 0
        total = 0
        epoch_loss = 0
        batch_count = 0
        epoch_entropy = 0
        epoch_conf = 0

        # Process in batches
        for i in range(0, len(out_train_augmented), batch_size):
          batch = out_train_augmented[i:i+batch_size]

          image_list = []
          for img in batch:
            image_list.append(img)

          images = torch.stack(image_list).to(device_classify)

          # Train adversarially: make model predict least likely class
          optimizer_classify.zero_grad()
          outputs = model_classify(images)
          
          # Get the least likely class (minimum probability) for each image
          _, fake_labels = torch.min(outputs.data, 1)  # get the least likely class
          
          # Compute loss with fake labels (adversarial training)
          loss = criterion_classify(outputs, fake_labels)
          loss.backward()
          optimizer_classify.step()

          # Accumulate graph metrics (per batch)
          epoch_loss += loss.item()
          batch_count += 1

          # Log out domain entropy and confidence
          with torch.no_grad():
            probs = torch.softmax(outputs, dim=1)
            epoch_entropy += (-torch.sum(probs * torch.log(probs + 1e-8), dim=1)).mean().item()
            epoch_conf += probs.max(dim=1)[0].mean().item()

          # Track what we're doing (for debugging)
          total += images.size(0)
          _, predicted = torch.min(outputs.data, 1)  # get the least likely class
          correct += (predicted == fake_labels).sum().item()

        scheduler_classify.step()
        current_acc = correct / total if total > 0 else 0.0

        # Average metrics of batch
        adv_ce_loss_history.append(epoch_loss / batch_count)
        adv_accuracy_history.append(correct / total)
        print(f"Out-domain epoch {epoch+1} → Adversarial accuracy: {current_acc:.4f}")

  except KeyboardInterrupt:
    print(" Moving to test")

  model_classify.eval()
  
  # Return wrapped model
  return ModelWrapper(model_classify, None, class_to_idx, num_classes)







def compute_accuracy(path_to_eval_folder, model):
  # Extract models and class info from wrapper
  model_classify = model.model_classify
  class_to_idx = model.class_to_idx
  num_classes = model.num_classes
  
  eval_data_pil = load_data(path_to_eval_folder, True, class_to_idx)
  # Transform PIL images to tensors
  eval_transform = get_transforms(False)
  eval_data = []
  for img, label in eval_data_pil:
      transformed_img = eval_transform(img)
      eval_data.append((transformed_img, label))

  device_classify = torch.device( "cuda" if torch.cuda.is_available() else "cpu")
  model_classify = model_classify.to(device_classify)
  model_classify.eval()  # Set to evaluation mode

  # Determine true category label (0 for in-domain, 1 for out-domain)
  true_category = 0 if "in-domain" in path_to_eval_folder else 1

  # Evaluate model
  correct = 0
  total = 0

  # Process in batches (more efficient)
  batch_size = 32
  with torch.no_grad():  # Don't compute gradients during evaluation
    for i in range(0, len(eval_data), batch_size):
      batch = eval_data[i:i+batch_size]
      # Expand batch processing: extract images and labels with extra checks, convert to tensors, and move to the required devices.
      image_list = []
      label_list = []
      for img, label in batch:
          image_list.append(img)
          label_list.append(label)
      images = torch.stack(image_list).to(device_classify)
      labels = torch.tensor(label_list).to(device_classify)
      
      # First, determine if each image is in-domain or out-domain
      outputs = model_classify(images)
      _, predicted = torch.max(outputs.data, 1) # get the predicted class
      
      # Separate images and labels based on in/out domain prediction

      
      

      # For in-domain images: classify them
      images_to_classify_tensor = images 
      labels_to_classify_tensor = labels 
      
      outputs = model_classify(images_to_classify_tensor)
      _, predicted = torch.max(outputs.data, 1)

      correct += (predicted == labels_to_classify_tensor).sum().item()
      total += len(images)
        

          
    

  # Calculate accuracies
  accuracy = correct / total

  return accuracy







import time
time_start = time.time()

# Graph code
train_acc_history = []
in_ce_loss_history = []
out_entropy_history = []
out_conf_history = []
adv_accuracy_history = []
adv_ce_loss_history = []

model = learn('./A4data/in-domain-train', './A4data/out-domain-train')

#Plot training accuracy over epochs
epochs = range(1, len(train_acc_history) + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_acc_history, marker='o', label="Train Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("In-Domain training Accuracy Over Epochs")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("/home/student/3105/A4files/train_accuracy.png")
plt.close()

#Plot CE loss per epoch
plt.figure(figsize=(8,5))
plt.plot(epochs, in_ce_loss_history, marker='o', label="Cross-Entropy Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("In-Domain CE Loss Over Epochs")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("/home/student/3105/A4files/CE_Loss.png")
plt.close()

# Plot entropy loss per epoch (non negated)
plt.figure(figsize=(8,5))
plt.plot(epochs, out_entropy_history, marker='o', label="Out-Domain Entropy")
plt.xlabel("Epoch")
plt.ylabel("Entropy")
plt.title("Out-Domain Entropy Over Epochs")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("/home/student/3105/A4files/Entropy_loss.png")
plt.close()

# Plot out domain confidence
plt.figure(figsize=(8,5))
plt.plot(epochs, out_conf_history, marker='o', label="Out-Domain Confidence")
plt.xlabel("Epoch")
plt.ylabel("Avg Max Softmax Probability")
plt.title("Out-Domain Confidence Over Epochs")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("/home/student/3105/A4files/domain_confidence.png")
plt.close()

# The following 2 are broken for the time being
# Plot adversarial training accuracy
# plt.figure(figsize=(8,5))
# plt.plot(epochs, adv_accuracy_history, marker='o', label="Adversarial Accuracy")
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.title("Adversarial Training Accuracy Over Epochs")
# plt.grid(alpha=0.3)
# plt.tight_layout()
# plt.savefig("/home/student/3105/A4files/adv_accuracy.png")
# plt.close()

# # Plot adversarial training CE loss
# plt.figure(figsize=(8,5))
# plt.plot(epochs, adv_ce_loss_history, marker='o', label="Adversarial CE Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Cross-Entropy Loss")
# plt.title("Adversarial Training CE Loss Over Epochs")
# plt.grid(alpha=0.3)
# plt.tight_layout()
# plt.savefig("/home/student/3105/A4files/adv_ce_loss.png")
# plt.close()


# Test on in-domain eval
in_acc = compute_accuracy('./A4data/in-domain-eval', model)
print(f"In-domain accuracy: {in_acc:.4f}")

# Test on out-domain eval
out_acc = compute_accuracy('./A4data/out-domain-eval', model)
print(f"Out-domain accuracy: {out_acc:.4f}")




time_end = time.time()
print(f"Time taken: {(time_end - time_start)/60:.2f} minutes")