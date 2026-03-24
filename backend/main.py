from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io
import base64
import os
from dotenv import load_dotenv

load_dotenv()

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5500")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL, "*"], # Allow frontend URL explicitly, and keep * for fallback or just the frontend URL depending on strictness. The user wants it to reflect the variable. Let's use [FRONTEND_URL] or ["*"] 
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
MODEL_PATH = "model/dr_model.keras"
model = None

try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

def get_gradcam_heatmap(img_array, model, last_conv_layer_name):
    x = model.inputs[0]
    last_conv_output = None
    for layer in model.layers:
        x = layer(x)
        if layer.name == last_conv_layer_name:
            last_conv_output = x
            
    grad_model = tf.keras.models.Model(model.inputs, [last_conv_output, x])
    img_tensor = tf.convert_to_tensor(img_array)

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_tensor)
        class_channel = preds[:, 0]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    if grads is None:
        return np.zeros((img_array.shape[1], img_array.shape[2]))

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    max_heat = tf.math.reduce_max(heatmap)
    if max_heat == 0:
        return heatmap.numpy()
    heatmap /= max_heat
    return heatmap.numpy()

def overlay_gradcam(img_bgr, heatmap, alpha=0.4):
    heatmap_resized = cv2.resize(heatmap, (img_bgr.shape[1], img_bgr.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    jet_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img_bgr, 1-alpha, jet_heatmap, alpha, 0)
    return superimposed_img

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not loaded properly."}
        
    # Read image
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert('RGB')
    
    # Preprocess
    img_resized = img.resize((128, 128))
    img_array = np.array(img_resized)
    img_batch = np.expand_dims(img_array, axis=0) # Shape: (1, 128, 128, 3)
    
    # We do NOT manually divide by 255 if we have Rescaling layer in model
    # Wait, does Rescaling layer do it? Yes, we added layers.Rescaling(1./255).
    # Prediction
    preds = model.predict(img_batch)
    prob = float(preds[0][0])
    
    # Classification logic
    if prob < 0.4:
        label = "No DR"
    elif prob <= 0.6:
        label = "Doctor Review Required"
    else:
        label = "Referable DR"
        
    # Grad-CAM
    try:
        heatmap = get_gradcam_heatmap(img_batch, model, 'last_conv_layer')
        
        # Original image in BGR for OpenCV
        img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        gradcam_img = overlay_gradcam(img_bgr, heatmap)
        
        # Convert back to RGB and base64
        gradcam_rgb = cv2.cvtColor(gradcam_img, cv2.COLOR_BGR2RGB)
        gradcam_pil = Image.fromarray(gradcam_rgb)
        
        buffered = io.BytesIO()
        gradcam_pil.save(buffered, format="JPEG")
        gradcam_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
    except Exception as e:
        print(f"GradCAM generation error: {e}")
        gradcam_b64 = None

    return {
        "prediction_label": label,
        "confidence_score": prob,
        "gradcam_image_base64": gradcam_b64
    }
