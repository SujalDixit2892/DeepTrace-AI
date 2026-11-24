from pathlib import Path
import math
import cv2
import numpy as np
from PIL import Image

MAX_BYTES = 100 * 1024 * 1024
MAX_SECONDS = 60
ALLOWED = {'.mp4', '.mov', '.webm'}

def stamp(seconds):
    s = max(0, int(seconds))
    return f'{s // 60:02d}:{s % 60:02d}'

def analyze_video(path, predict_rgb, threshold=0.5, margin=0.05, progress=None):
    p = Path(path)
    if not p.is_file(): raise ValueError('Uploaded video was not found.')
    if p.suffix.lower() not in ALLOWED: raise ValueError('Use MP4, MOV, or WEBM.')
    if p.stat().st_size > MAX_BYTES: raise ValueError('Video exceeds the 100 MB limit.')
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened(): raise ValueError('Could not open video; it may be corrupt or use an unsupported codec.')
    scores, points, previews = [], [], []
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0 or total <= 0: raise ValueError('Could not read video duration or frame rate.')
        duration = total / fps
        if duration > MAX_SECONDS: raise ValueError('Video exceeds the 60-second limit.')
        step = max(1, round(fps))
        index = 0
        while len(scores) < 60:
            ok, bgr = cap.read()
            if not ok: break
            if index % step == 0:
                t = index / fps
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                score = predict_rgb(rgb)
                scores.append(score)
                points.append({'timestamp': round(t, 2), 'p_fake': round(score, 6)})
                if len(previews) < 6:
                    thumb = Image.fromarray(rgb); thumb.thumbnail((360, 240))
                    previews.append((thumb, f'{stamp(t)} — P(fake) {score:.1%}'))
                if progress is not None: progress(min(.99, index / max(total, 1)), desc=f'Analyzing frame {len(scores)}')
            index += 1
        if not scores: raise ValueError('No readable video frames were found.')
    finally:
        cap.release()
    avg = float(np.mean(scores))
    above = 100 * sum(x >= threshold for x in scores) / len(scores)
    if avg >= threshold + margin and above >= 65:
        label, explanation = 'FAKE', 'Average score and a substantial majority of sampled frames support a fake classification.'
    elif avg <= threshold - margin and above <= 35:
        label, explanation = 'REAL', 'Average score and sampled-frame agreement support a real classification.'
    else:
        label, explanation = 'INCONCLUSIVE', 'Sampled frames show mixed or borderline evidence; do not treat this as definitive.'
    hi = int(np.argmax(scores))
    if progress is not None: progress(1.0, desc='Video analysis complete')
    return {'assessment': label, 'explanation': explanation, 'average_fake_score': round(avg, 6),
            'average_real_score': round(1-avg, 6), 'frames_above_threshold_percent': round(above, 2),
            'sampled_frames': len(scores), 'duration_seconds': round(duration, 2), 'fps': round(fps, 3),
            'sampling_rate': 'approximately 1 FPS', 'highest_frame_timestamp': stamp(points[hi]['timestamp']),
            'highest_frame_fake_score': round(max(scores), 6), 'threshold': threshold,
            'uncertainty_margin': margin, 'chart_data': points, 'previews': previews}
