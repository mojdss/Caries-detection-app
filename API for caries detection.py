import io
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
import threading
import warnings
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Suppress specific warnings related to experimental features
warnings.filterwarnings("ignore", category=UserWarning, message=".*experimental feature.*")

# Define the path to the model
model_path = r'd:\MY CV\PROJECTS\Caries detection app\models with netdata\vgg_model.pthd:\paper code\models with netdata\vgg_model.pth'  # Update this path as needed

# Define a Pydantic model for the response
class PredictionResponse(BaseModel):
    prediction: str
    confidence: float

# Initialize FastAPI app
app = FastAPI(
    title="VGG Model API",
    description="API for detecting caries in panoramic images using a VGG model.",
    version="1.0.0",
)

# Define the VGG model class
class VGGModel:
    def __init__(self, model_path):
        self.model = models.vgg16(pretrained=True)
        self.model.classifier[6] = nn.Linear(self.model.classifier[6].in_features, 2)  # Adjust for 2 classes
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def predict(self, image: Image.Image):
        # Convert grayscale images to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(image)
            _, predicted = torch.max(outputs, 1)
            confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted].item()
            class_idx = predicted.item()
            logger.debug(f'Prediction index: {class_idx}, Confidence: {confidence}')
            return class_idx, confidence

vgg_model = VGGModel(model_path)

@app.get("/", response_class=HTMLResponse)
async def index():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Caries Detection</title>
        <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css">
        <style>
            body { font-family: Arial, sans-serif; background-color: #f8f9fa; margin: 0; }
            .container { max-width: 800px; margin: 40px auto; padding: 20px; background: white; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
            h1 { color: #343a40; }
            #result { margin-top: 20px; }
            .alert { display: none; }
            .btn-primary { background-color: #007bff; border-color: #007bff; }
            .btn-primary:hover { background-color: #0056b3; border-color: #004a9b; }
            .card { border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
            .card-body { padding: 20px; }
            .card-title { font-size: 1.25rem; }
            .card-text { font-size: 1rem; }
            .spinner-border { width: 3rem; height: 3rem; border-width: 0.4em; }
            #resultCard { background-color: #007bff; color: white; padding: 20px; border-radius: 8px; margin-top: 20px; display: none; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="text-center">Caries Detection API</h1>
            <div class="card">
                <div class="card-body">
                    <h5 class="card-title">Upload an Image</h5>
                    <form id="uploadForm">
                        <input type="file" id="fileInput" class="form-control-file" accept=".jpg,.jpeg,.png" required>
                        <button type="submit" class="btn btn-primary mt-3">Upload and Predict</button>
                    </form>
                    <div id="loading" class="mt-3" style="display: none;">
                        <div class="spinner-border" role="status">
                            <span class="sr-only">Loading...</span>
                        </div>
                        <p class="mt-2">Processing your image...</p>
                    </div>
                    <div id="resultCard">
                        <h2 id="prediction" class="card-title"></h2>
                        <p id="confidence" class="card-text"></p>
                    </div>
                    <div id="error" class="alert alert-danger"></div>
                </div>
            </div>
        </div>
        <script src="https://code.jquery.com/jquery-3.5.1.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.5.3/dist/umd/popper.min.js"></script>
        <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
        <script>
            document.getElementById('uploadForm').addEventListener('submit', async function(event) {
                event.preventDefault();
                const fileInput = document.getElementById('fileInput');
                const file = fileInput.files[0];
                const formData = new FormData();
                formData.append('file', file);

                const loadingDiv = document.getElementById('loading');
                const resultCard = document.getElementById('resultCard');
                const predictionText = document.getElementById('prediction');
                const confidenceText = document.getElementById('confidence');
                const errorDiv = document.getElementById('error');

                errorDiv.style.display = 'none';
                resultCard.style.display = 'none';
                loadingDiv.style.display = 'block';

                try {
                    const response = await fetch('/predict/', {
                        method: 'POST',
                        body: formData,
                    });

                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }

                    const result = await response.json();
                    loadingDiv.style.display = 'none';
                    resultCard.style.display = 'block';
                    predictionText.textContent = `Prediction: ${result.prediction}`;
                    confidenceText.textContent = `Confidence: ${result.confidence.toFixed(2)}`;
                } catch (error) {
                    loadingDiv.style.display = 'none';
                    errorDiv.textContent = `Error: ${error.message}`;
                    errorDiv.style.display = 'block';
                    console.error('Error:', error);  // Log the error to the console
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/predict/", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read()))
        # Convert grayscale images to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        class_idx, confidence = vgg_model.predict(image)
        class_names = ["Caries", "Non Caries"]
        prediction = class_names[class_idx]
        logger.info(f'Prediction: {prediction}, Confidence: {confidence}')
        return PredictionResponse(prediction=prediction, confidence=confidence)
    except Exception as e:
        logger.error(f'Error: {e}', exc_info=True)  # Log the full traceback
        raise HTTPException(status_code=500, detail="Model prediction failed. Please try again.")

# Run the server programmatically
def run_server():
    uvicorn.run("API for caries detection:app", host="127.0.0.1", port=8001, reload=False)


# Start server in a thread
if __name__ == "__main__":
    server_thread = threading.Thread(target=run_server)
    server_thread.start()
    print("Server is running on http://127.0.0.1:8001/")
    try:
        server_thread.join()
    except KeyboardInterrupt:
        print("Server stopped manually.")
