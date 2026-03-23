# Diabetic Retinopathy Detection System

This is a complete, simple, and functional AI-based Diabetic Retinopathy Detection System.

## Features
- **AI Model**: A Deep Learning CNN built using TensorFlow/Keras that classifies retinal images as `No DR`, `Doctor Review Required`, or `Referable DR`.
- **Explainability**: Generates a Grad-CAM heatmap overlay to show which regions of the eye influenced the model's decision.
- **Backend**: FastAPI server that loads the model and exposes a RESTful API.
- **Frontend**: Clean and simple HTML/JS UI for easy image upload and analysis viewing.

## Setup Instructions

### 1. Requirements
Ensure Python 3.10+ is installed on your Windows machine.

### 2. Create Virtual Environment
Open PowerShell in the `dr-detection` folder:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Data Preparation
We use the MedMNIST (RetinaMNIST) binary dataset. Execute the preparation script to automatically download and process the data:
```powershell
python prepare_data.py
```

### 5. Train the Model
The model trains automatically to reach >65% accuracy, adjusting and retrying if necessary.
```powershell
python train.py
```

### 6. Run the Backend
Start the FastAPI server (it runs on port 8000 by default):
```powershell
cd backend
uvicorn main:app --reload
```

### 7. View the Frontend
Simply open the `index.html` file inside the `frontend/` folder in your browser.
Click "Select Image" and choose an image from `data/val/1_dr` or `data/val/0_no_dr`.

## Deployment (Docker/Cloud)
To deploy this system to the cloud:
1. Containerize the backend by creating a `Dockerfile` that `pip installs` the requirements and runs `uvicorn backend.main:app --host 0.0.0.0 --port 80`.
2. Host the frontend statically on Vercel, Netlify, or AWS S3.
3. Ensure the frontend `fetch` request points to your deployed backend URL instead of `localhost:8000`.
