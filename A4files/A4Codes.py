# TODO: Add necessary imports
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
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
  
  for _ in range(6):
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


  # create the categorisation trainingdata 
  # 0 -> in-domain, 1 -> out-domain
  categorise_data_train = []
  for (image, _) in in_train_augmented:
    categorise_data_train.append((image, 0))
  
  for image in out_train_augmented:
    categorise_data_train.append((image, 1))

  print("data loaded")





  #============================= Categorisation Model =======================================

  # create the categorisation model
  device_categorise = torch.device( "cuda" if torch.cuda.is_available() else "cpu")
  model_categorise = resnet18(weights=None) # not allowed to use pretrained model
  model_categorise.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model_categorise.fc.in_features, 2)
  )
  model_categorise = model_categorise.to(device_categorise)
  
  criterion_categorise = nn.CrossEntropyLoss()
  optimizer_categorise = optim.Adam(model_categorise.parameters(), lr=0.002, weight_decay=0.0001)
  scheduler_categorise = optim.lr_scheduler.StepLR(optimizer_categorise, step_size=16, gamma=0.5)
  
  model_categorise.train()
  # Convergence check parameters
  best_acc_categorise = 0.0 # best accuracy so far
  patience_categorise = 10  # amount of epochs to wait for improvement
  no_improve_categorise = 0 # amount of epochs without improvement
  min_improvement = 0.002  # Only count improvements > 0.2% as meaningful


  batch_size = 64

  try:
    # epoch num is dynamic; stops when convergence is reached
    for epoch in range(100):
      # Shuffle data
      import random
      random.shuffle(categorise_data_train)
      correct = 0
      total = 0


      # Process in batches
      for i in range(0, len(categorise_data_train), batch_size):
        batch = categorise_data_train[i:i+batch_size]

        image_list = []
        label_list = []

        for img, label in batch:
            image_list.append(img)
            label_list.append(label)

        images = torch.stack(image_list).to(device_categorise)
        labels = torch.tensor(label_list).to(device_categorise)


        # Train
        optimizer_categorise.zero_grad()
        outputs = model_categorise(images)
        loss = criterion_categorise(outputs, labels)
        loss.backward()
        optimizer_categorise.step()


        total += labels.size(0)
        _, predicted = torch.max(outputs.data, 1) # get the predicted class
        correct += (predicted == labels).sum().item() # count the correct predictions

      scheduler_categorise.step()
      current_acc = correct / total
      
      # Convergence check with meaningful improvement threshold
      improvement = current_acc - best_acc_categorise
      if improvement > min_improvement:
        best_acc_categorise = current_acc
        no_improve_categorise = 0
        print(f"epoch {epoch+1} → New best accuracy: {best_acc_categorise:.4f} (+{improvement:.4f})")
      else:
        no_improve_categorise += 1
        print(f"epoch {epoch+1} → No improvement for {no_improve_categorise} epochs (best: {best_acc_categorise:.4f}, current: {current_acc:.4f})")
      
      # Early stop if accuracy is very high (likely overfitting)
      if current_acc > 0.98:
        print(f"\nepoch {epoch+1} → High accuracy reached ({current_acc:.4f}), stopping to prevent overfitting.")
        break
        
      if no_improve_categorise >= patience_categorise:
        print(f"\nepoch {epoch+1} → Convergence reached! No improvement for {patience_categorise} epochs. Stopping training.")
        print(f"epoch {epoch+1} → Best accuracy: {best_acc_categorise:.4f}")
        break


  except KeyboardInterrupt:
    print(" Categorisation model trained")

  print(" Moving to classify")





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
  scheduler_classify = optim.lr_scheduler.StepLR(optimizer_classify, step_size=8, gamma=0.5)

  # Use the pre-augmented in-domain data
  model_classify.train()
  # Convergence check parameters
  best_acc_classify = 0.0 # best accuracy so far
  patience_classify = 6  # amount of epochs to wait for improvement
  no_improve_classify = 0 # amount of epochs without improvement
  min_improvement_classify = 0.002  # Only count improvements > 0.2% as meaningful

  try:
    for epoch in range(100):
      # Shuffle data
      import random
      random.shuffle(in_train_augmented)
      correct = 0
      total = 0


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

        # Calculate accuracy training
        total += labels.size(0)
        _, predicted = torch.max(outputs.data, 1) # get the predicted class
        correct += (predicted == labels).sum().item()

      scheduler_classify.step()
      current_acc = correct / total
      
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

  except KeyboardInterrupt:
    print(" Moving to test")

  model_classify.eval()
  model_categorise.eval()
  
  # Return wrapped model
  return ModelWrapper(model_classify, model_categorise, class_to_idx, num_classes)







def compute_accuracy(path_to_eval_folder, model):
  # Extract models and class info from wrapper
  model_classify = model.model_classify
  model_categorise = model.model_categorise
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
  # Conditional accuracy: accuracy for correctly categorized images
  conditional_correct = 0
  conditional_total = 0

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
      images = torch.stack(image_list).to(device_categorise)
      labels = torch.tensor(label_list).to(device_classify)
      
      # First, determine if each image is in-domain or out-domain
      categorise_outputs = model_categorise(images)
      _, predicted_categorise = torch.max(categorise_outputs.data, 1) # get the predicted class
      
      # Track categorization accuracy
      categorise_total += len(images)
      categorise_correct += (predicted_categorise == true_category).sum().item() # count the correct predictions
      
      # Separate images and labels based on in/out domain prediction
      images_to_classify = []
      labels_to_classify = []
      indices_to_guess = []
      # Track which indices are actually the correct category for conditional accuracy
      correct_category_indices_classify = []
      correct_category_indices_guess = []


      for idx in range(len(images)):
        if predicted_categorise[idx] == 0:  # In-domain
          images_to_classify.append(images[idx])
          labels_to_classify.append(labels[idx])
          # Track if this was correctly categorized as 0
          if true_category == 0:
            correct_category_indices_classify.append(len(images_to_classify) - 1)
        else:  # Out-domain
          indices_to_guess.append(idx)
          # Track if this was correctly categorized as 1
          if true_category == 1:
            correct_category_indices_guess.append(len(indices_to_guess) - 1)
      
      
      # For in-domain images: classify them
      if len(images_to_classify) > 0:
        images_to_classify_tensor = torch.stack(images_to_classify).to(device_classify)
        labels_to_classify_tensor = torch.tensor(labels_to_classify).to(device_classify)

        outputs = model_classify(images_to_classify_tensor)
        _, predicted = torch.max(outputs.data, 1)

        correct += (predicted == labels_to_classify_tensor).sum().item()
        total += len(images_to_classify)
        
        # Conditional accuracy: only for correctly categorized as 0
        if len(correct_category_indices_classify) > 0:
          conditional_predicted = predicted[correct_category_indices_classify]
          conditional_labels = labels_to_classify_tensor[correct_category_indices_classify]
          conditional_correct += (conditional_predicted == conditional_labels).sum().item()
          conditional_total += len(correct_category_indices_classify)

          
      
      # For out-domain images: just guess randomly
      # For out-domain images: guess the least likely class (intentionally wrong)
      if len(indices_to_guess) > 0:
        # Stack images that need guessing
        # Prepare tensors for out-domain images to be guessed
        images_to_guess_list = []
        labels_to_guess_list = []

        for idx in indices_to_guess:
          images_to_guess_list.append(images[idx])
          labels_to_guess_list.append(labels[idx])

        images_to_guess = torch.stack(images_to_guess_list).to(device_classify)
        labels_to_guess = torch.tensor(labels_to_guess_list).to(device_classify)
        

        # Get classification probabilities
        guess_outputs = model_classify(images_to_guess)
        guess_probs = torch.softmax(guess_outputs, dim=1)
        
        # Guess the least likely class (argmin instead of argmax)
        _, least_likely = torch.min(guess_probs, dim=1)
        
        # Check accuracy (will be intentionally low)
        correct += (least_likely == labels_to_guess).sum().item()
        total += len(indices_to_guess)
        
        # Conditional accuracy: only for correctly categorized as 1
        if len(correct_category_indices_guess) > 0:
          conditional_least_likely = least_likely[correct_category_indices_guess]
          conditional_labels_guess = labels_to_guess[correct_category_indices_guess]
          conditional_correct += (conditional_least_likely == conditional_labels_guess).sum().item()
          conditional_total += len(correct_category_indices_guess)  



  # Calculate accuracies
  accuracy = correct / total
  categorise_accuracy = categorise_correct / categorise_total
  conditional_accuracy = conditional_correct / conditional_total if conditional_total > 0 else 0.0
  
  
  print(f"\nCategorization accuracy: {categorise_accuracy:.4f}")
  print(f"Conditional accuracy (correctly categorized only): {conditional_accuracy:.4f}")

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