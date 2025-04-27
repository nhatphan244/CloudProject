import torch
import torchvision.transforms as transforms
from PIL import Image
import argparse
import os
from torchvision import models
import torch.nn as nn
import json

def load_model(model_path):
    """
    Load the trained model
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        model: Loaded PyTorch model
        class_names: List of class names
    """
    # Check if CUDA is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint['class_names']
    num_classes = len(class_names)
    
    # Initialize model
    model = models.efficientnet_b0(weights='IMAGENET1K_V1')
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, class_names, device

def predict_image(model, image_path, transform, class_names, device):
    """
    Predict monster from an image
    
    Args:
        model: Trained PyTorch model
        image_path: Path to the image
        transform: Transform to apply to the image
        class_names: List of class names
        device: Device to run inference on
        
    Returns:
        predicted_class: Predicted monster name
        confidence: Confidence score
    """
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
    # Get top 3 predictions
    values, indices = torch.topk(probabilities, 3)
    
    results = []
    for i in range(3):
        results.append({
            'monster': class_names[indices[0][i].item()],
            'confidence': values[0][i].item() * 100
        })
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Monster Hunter Monster Classifier')
    parser.add_argument('--image', required=True, help='Path to the image file')
    parser.add_argument('--model', default='./model/monster_hunter_classifier_v3.pth', help='Path to the trained model')
    args = parser.parse_args()
    
    # Define transform for inference
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load model
    model, class_names, device = load_model(args.model)
    
    # Check if the image path exists
    if not os.path.exists(args.image):
        print(f"Error: Image file '{args.image}' not found")
        return
    
    # Predict monster
    results = predict_image(model, args.image, transform, class_names, device)

    ans = {}
    
    # Print results
    print("\nMonster Hunter Classification Results:")
    print("=" * 40)
    for i, result in enumerate(results):
        print(f"{i+1}. {result['monster']} - {result['confidence']:.2f}% confidence")


if __name__ == "__main__":
    main()