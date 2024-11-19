from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models

app = Flask(__name__)

# Define the classes for prediction
classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Initialize the ResNet50 model with no pretrained weights
model = models.resnet50(weights=None)

# Modify the final layer to match the number of classes
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(classes))

# Correctly set the path to the model file, ensuring it works in different environments
model_path = os.path.join(os.path.dirname(__file__), 'model.pth')

# Load the model's state dict
try:
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# Set the model to evaluation mode
model.eval()

# Define the device to run the model on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define the image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join('uploads', filename)
            file.save(file_path)
            
            prediction, confidence = predict_tumor_type(file_path)
            
            return render_template('result.html', prediction=prediction, confidence=confidence)
    
    return render_template('index.html')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def predict_tumor_type(image_path):
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    image = image.to(device)
    
    # Make prediction with the model
    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    # Capitalize the tumor type for display
    predicted_class = classes[predicted.item()].capitalize()
    
    return predicted_class, confidence.item() * 100

if __name__ == '__main__':
    # Create the uploads directory if it doesn't exist
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    
    # Run the Flask app
    app.run(debug=True)
