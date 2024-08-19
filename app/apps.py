from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models

app = Flask(__name__)

classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

model = models.resnet50(weights=None)

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(classes))

model.load_state_dict(torch.load(r"C:\Users\dcken\Documents\DPA\model.pth", map_location=torch.device('cpu')))

model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

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
            
            prediction = predict_tumor_type(file_path)
            
            return render_template('result.html', prediction=prediction)
    
    return render_template('index.html')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def predict_tumor_type(image_path):
    image = Image.open(image_path)
    
    image = image.convert('RGB')
    
    image = transform(image).unsqueeze(0)
    
    image = image.to(device)
    
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    
    return classes[predicted.item()]

if __name__ == '__main__':
    app.run(debug=True)
