from fastapi.testclient import TestClient
from backend.main import app
import os

client = TestClient(app)

def test_predict():
    # Pick a sample image from val set
    img_dir = "data/val/1_dr"
    if not os.path.exists(img_dir):
        print("Data directory not found for tests.")
        return
        
    files = os.listdir(img_dir)
    if not files:
        print("No files found to test.")
        return
        
    test_img = os.path.join(img_dir, files[0])
    
    with open(test_img, "rb") as f:
        response = client.post("/predict", files={"file": ("test.jpg", f, "image/jpeg")})
        
    print(f"Status code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Prediction: {data.get('prediction_label')}")
        print(f"Confidence: {data.get('confidence_score')}")
        print(f"GradCAM generated: {'yes' if data.get('gradcam_image_base64') else 'no'}")
    else:
        print(f"Error: {response.text}")

if __name__ == "__main__":
    test_predict()
