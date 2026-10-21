import json
import sys
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL = "model/crop_disease_model.keras"
CLASSES = "model/class_names.json"

if len(sys.argv) != 2:
    print("Usage: python predict.py path/to/image.jpg")
    raise SystemExit(1)

model = load_model(MODEL)
with open(CLASSES, encoding="utf-8") as f:
    names = json.load(f)

img = Image.open(sys.argv[1]).convert("RGB")
img = img.resize(tuple(model.input_shape[1:3]))
x = preprocess_input(np.expand_dims(np.asarray(img, dtype=np.float32), 0))
p = model.predict(x, verbose=0)[0]
i = int(np.argmax(p))

print("Prediction:", names[i])
print("Confidence:", f"{p[i] * 100:.2f}%")
