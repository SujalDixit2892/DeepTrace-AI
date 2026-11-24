import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '2')

from pathlib import Path
from threading import Lock
import cv2
import numpy as np
from PIL import Image

class ModelManager:
    def __init__(self, model_path='models/final_deepfake_model.keras'):
        self.model_path = Path(model_path)
        self.model = None
        self._load_lock = Lock()
        self._predict_lock = Lock()

    def load(self):
        with self._load_lock:
            if self.model is not None:
                return self.model
            if not self.model_path.is_file():
                raise FileNotFoundError(f'Model not found: {self.model_path.resolve()}')
            import tensorflow as tf
            try:
                tf.config.set_visible_devices([], 'GPU')
            except RuntimeError:
                pass
            try:
                tf.config.threading.set_intra_op_parallelism_threads(2)
                tf.config.threading.set_inter_op_parallelism_threads(1)
            except RuntimeError:
                pass
            self.model = tf.keras.models.load_model(str(self.model_path), compile=False)
            if tuple(self.model.input_shape[-3:]) != (224, 224, 3):
                raise ValueError(f'Unexpected model input: {self.model.input_shape}')
            if self.model.output_shape[-1] != 1:
                raise ValueError(f'Expected one sigmoid output, got {self.model.output_shape}')
            return self.model

    @staticmethod
    def preprocess_rgb(rgb):
        arr = np.asarray(rgb)
        if arr.ndim != 3 or arr.shape[2] not in (3, 4):
            raise ValueError('Expected an RGB image.')
        arr = arr[:, :, :3].astype(np.uint8, copy=False)
        # Notebook input is 224x224 RGB; do not divide by 255 a second time.
        return cv2.resize(arr, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32)

    def predict_rgb(self, rgb):
        model = self.load()
        batch = np.expand_dims(self.preprocess_rgb(rgb), 0)
        with self._predict_lock:
            output = model(batch, training=False)
            score = float(np.asarray(output).reshape(-1)[0])
        if not np.isfinite(score):
            raise RuntimeError('Model returned an invalid prediction.')
        return float(np.clip(score, 0.0, 1.0))

    def predict_image_path(self, path):
        with Image.open(path) as im:
            return self.predict_rgb(np.asarray(im.convert('RGB')))

def classify_score(score, threshold=0.5, margin=0.05):
    if score >= threshold + margin:
        return {'label': 'FAKE', 'explanation': 'The model score is above the fake decision boundary.'}
    if score <= threshold - margin:
        return {'label': 'REAL', 'explanation': 'The model score is below the fake decision boundary.'}
    return {'label': 'INCONCLUSIVE', 'explanation': 'The score is close to the decision boundary; treat it cautiously.'}

def image_metadata(path):
    p = Path(path)
    size = p.stat().st_size
    if size > 20 * 1024 * 1024:
        raise ValueError('Image exceeds the 20 MB limit.')
    with Image.open(p) as im:
        fmt = (im.format or p.suffix.lstrip('.')).upper()
        w, h = im.size
        if fmt not in {'JPEG', 'JPG', 'PNG', 'WEBP'}:
            raise ValueError('Use JPG, PNG, or WEBP images.')
        if w < 32 or h < 32 or w * h > 40_000_000:
            raise ValueError('Image dimensions are invalid or exceed 40 megapixels.')
    return {'filename': p.name, 'format': fmt, 'file_size_mb': round(size / 1048576, 3),
            'original_width': w, 'original_height': h, 'resized_for_model': (w, h) != (224, 224)}
