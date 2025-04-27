import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tqdm import tqdm
import random
import time

# Set random seeds for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Check if CUDA is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class MonsterHunterDataset(Dataset):
    def __init__(self, data_dir, transform=None, split='train'):
        """
        MonsterHunterDataset constructor
        
        Args:
            data_dir (str): Directory containing monster subdirectories
            transform (callable, optional): Transform to be applied to images
            split (str): 'train', 'val', or 'test'
        """
        self.data_dir = data_dir
        self.transform = transform
        self.split = split
        
        # Get all monster classes (folder names)
        self.classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        self.classes.sort()  # Sort for consistency
        print(f"Found {len(self.classes)} monster classes")
        
        # Create class to index mapping
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Collect all image paths and their labels
        self.image_paths = []
        self.labels = []
        
        for cls_name in self.classes:
            cls_dir = os.path.join(data_dir, cls_name)
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')):
                    self.image_paths.append(os.path.join(cls_dir, img_name))
                    self.labels.append(self.class_to_idx[cls_name])
        
        # Shuffle and split data
        indices = list(range(len(self.image_paths)))
        random.shuffle(indices)
        
        if split == 'train':
            # Use 70% for training
            indices = indices[:int(0.7 * len(indices))]
        elif split == 'val':
            # Use 15% for validation
            indices = indices[int(0.7 * len(indices)):int(0.85 * len(indices))]
        else:  # 'test'
            # Use 15% for testing
            indices = indices[int(0.85 * len(indices)):]
            
        self.image_paths = [self.image_paths[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
    
    def get_classes(self):
        return self.classes

# Define transforms for data augmentation and normalization
def get_transforms():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),  # Resize and randomly crop to 224x224
        transforms.RandomHorizontalFlip(),   # Randomly flip horizontally
        transforms.RandomRotation(10),       # Randomly rotate by up to 10 degrees
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Adjust color properties
        transforms.ToTensor(),               # Convert to tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize(256),           # Resize the image to 256x256
        transforms.CenterCrop(224),       # Center crop to 224x224
        transforms.ToTensor(),            # Convert to tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize
    ])
    
    return train_transform, val_test_transform

def initialize_model(num_classes, feature_extract=True):
    """
    Initialize a pre-trained ResNet50 model for transfer learning
    
    Args:
        num_classes (int): Number of output classes
        feature_extract (bool): If True, only update the reshaped layer params
    
    Returns:
        model (torchvision.models): Modified ResNet model
    """
    # Load pre-trained model
    model = models.resnet50(weights='IMAGENET1K_V2')
    
    # Freeze parameters if feature extracting
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False
    
    # Change the final layer to match our number of classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),  # Add dropout for regularization
        nn.Linear(num_ftrs, num_classes)
    )
    
    return model

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25):
    """
    Train the model
    
    Args:
        model: PyTorch model
        dataloaders: Dictionary with 'train' and 'val' dataloaders
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        num_epochs: Number of epochs to train for
    
    Returns:
        model: Trained model
        history: Training history
    """
    since = time.time()
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    best_model_wts = model.state_dict()
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
                
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            for inputs, labels in tqdm(dataloaders[phase], desc=phase):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train' and scheduler is not None:
                scheduler.step()
                
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Record history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.cpu().numpy())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.cpu().numpy())
            
            # Save the best model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                
        print()
        
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, history

def evaluate_model(model, test_loader, criterion, class_names):
    """
    Evaluate the model on test data
    
    Args:
        model: Trained PyTorch model
        test_loader: DataLoader for test data
        criterion: Loss function
        class_names: List of class names
    """
    model.eval()
    test_loss = 0.0
    test_corrects = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item() * inputs.size(0)
            test_corrects += torch.sum(preds == labels.data)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_loss = test_loss / len(test_loader.dataset)
    test_acc = test_corrects.double() / len(test_loader.dataset)
    
    print(f'Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}')
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

def plot_training_history(history):
    """
    Plot training history
    
    Args:
        history: Dictionary containing training and validation metrics
    """
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def predict_monster(model, image_path, transform, class_names):
    """
    Predict monster from an image
    
    Args:
        model: Trained PyTorch model
        image_path: Path to the image
        transform: Transform to apply to the image
        class_names: List of class names
        
    Returns:
        predicted_class: Predicted monster name
        confidence: Confidence score
    """
    model.eval()
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, prediction = torch.max(probabilities, 1)
    
    predicted_idx = prediction.item()
    predicted_class = class_names[predicted_idx]
    confidence_score = confidence.item()
    
    return predicted_class, confidence_score

def main():
    # Set data directory
    data_dir = './data/monster_images'
    
    # Get transformations
    train_transform, val_test_transform = get_transforms()
    
    # Create datasets
    train_dataset = MonsterHunterDataset(data_dir, transform=train_transform, split='train')
    val_dataset = MonsterHunterDataset(data_dir, transform=val_test_transform, split='val')
    test_dataset = MonsterHunterDataset(data_dir, transform=val_test_transform, split='test')
    
    # Get class names
    class_names = train_dataset.get_classes()
    num_classes = len(class_names)
    print(f"Total number of classes: {num_classes}")
    
    # Create dataloaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }
    
    # Initialize model
    model = initialize_model(num_classes, feature_extract=True)
    model = model.to(device)
    
    # Loss function, optimizer and scheduler
    criterion = nn.CrossEntropyLoss()
    
    # Only optimize parameters that require gradients
    params_to_update = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params_to_update, lr=0.001)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Train the model
    model, history = train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=15)
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate the model
    print("\nEvaluating model on test data:")
    evaluate_model(model, test_loader, criterion, class_names)
    
    # Save the model
    os.makedirs('./model', exist_ok=True) 
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names
    }, './model/monster_hunter_classifier.pth')
    
    print("Model saved as './model/monster_hunter_classifier.pth'")
    
    # Example of using the model for prediction
    print("\nExample prediction:")
    # Replace with a path to a test image
    test_image_path = os.path.join(data_dir, class_names[0], os.listdir(os.path.join(data_dir, class_names[0]))[0])
    predicted_monster, confidence = predict_monster(model, test_image_path, val_test_transform, class_names)
    print(f"Predicted monster: {predicted_monster} with confidence {confidence*100:.2f}%")

if __name__ == "__main__":
    main()