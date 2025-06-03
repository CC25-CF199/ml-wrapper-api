from fastapi import FastAPI, File, UploadFile
from PIL import Image
from pydantic import BaseModel
import tensorflow as tf
import uvicorn
import numpy as np
import io

app = FastAPI()

BURN_DEGREE_IMG_SIZE = (224, 224)
BODY_PART_IMG_SIZE = (128, 128)
burn_degree_model = tf.keras.models.load_model("ml_model/best_model.h5")
body_part_model = tf.keras.models.load_model("ml_model/body_part_model.h5")
burn_degree_class_names = ['1st degree burn', '2nd degree burn', '3rd degree burn'] 
body_part_class_names = ['Perut', 'Telinga', 'Siku', 'Mata', 'Kaki', 'Tangan', 'Lutut', 'Leher', 'Hidung', 'Bahu']

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
  burn_degree_image = image.resize(BURN_DEGREE_IMG_SIZE)
  body_part_image = image.resize(BODY_PART_IMG_SIZE)

  # Convert img to array
  burn_degree_img_arr = np.array(burn_degree_image)
  body_part_img_arr = np.array(body_part_image)

  # Normalize image
  burn_degree_img_arr = burn_degree_img_arr / 255
  body_part_img_arr = body_part_img_arr /255

  # Expands batch dimension
  burn_degree_input_batch = np.expand_dims(burn_degree_img_arr, axis=0)
  body_part_input_batch = np.expand_dims(body_part_img_arr, axis=0)

  degree_prediction = burn_degree_model.predict(burn_degree_input_batch)
  body_part_prediction = body_part_model.predict(body_part_input_batch)
  degree_predicted_class = int(np.argmax(degree_prediction, axis=1)[0])
  body_part_predicted_class = int(np.argmax(body_part_prediction, axis=1)[0])
  burn_degree_confidence = float(np.max(degree_prediction))
  body_part_confidence = float(np.max(body_part_prediction))

  return {
    "predicted_body_part": body_part_class_names[body_part_predicted_class],
    "predicted_class_label": burn_degree_class_names[degree_predicted_class],
    "burn_degree_confidence": burn_degree_confidence,
    "body_part_confidence": body_part_confidence
  }
