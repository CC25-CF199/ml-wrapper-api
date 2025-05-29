from fastapi import FastAPI, File, UploadFile
from PIL import Image
from pydantic import BaseModel
import tensorflow as tf
import uvicorn
import numpy as np
import io

app = FastAPI()

model = tf.keras.models.load_model("ml_model/best_model.h5")
IMG_SIZE = (224, 224)
class_names = ['1st degree burn', '2nd degree burn', '3rd degree burn']

@app.get('/')
def main():
  return {
    "message": "ML Wrapper FastAPI"
  }

@app.post('/predict')
async def predict_image(file: UploadFile = File(...)):
  image_bytes = await file.read()
  
  # Resize Image using PIL
  image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
  image = image.resize(IMG_SIZE)

  # Convert img to array
  img_arr = np.array(image)

  # Normalize image
  img_arr = img_arr / 255

  # Expands batch dimension
  input_batch = np.expand_dims(img_arr, axis=0)

  prediction = model.predict(input_batch)
  predicted_class = int(np.argmax(prediction, axis=1)[0])
  confidence = float(np.max(prediction))

  return {
    "predicted_class_index": predicted_class,
    "predicted_class_label": class_names[predicted_class],
    "confidence": confidence
  }
