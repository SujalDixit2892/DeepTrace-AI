import numpy as np
from inference import ModelManager
m = ModelManager('models/final_deepfake_model.keras')
model = m.load()
print('Loaded:', model.input_shape, '->', model.output_shape)
print('Dummy P(fake):', round(m.predict_rgb(np.zeros((224,224,3), dtype=np.uint8)), 6))
