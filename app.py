import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '2')

from pathlib import Path
import json
import tempfile
import gradio as gr
from gradio.themes import Soft
import pandas as pd
from inference import ModelManager, classify_score, image_metadata
from video_utils import analyze_video

MODEL_PATH = os.getenv('MODEL_PATH', 'models/final_deepfake_model.keras')
manager = ModelManager(MODEL_PATH)
try:
    manager.load(); MODEL_READY = True; MODEL_STATUS = 'Model loaded — CPU inference ready.'
except Exception as exc:
    MODEL_READY = False; MODEL_STATUS = f'Model unavailable: {exc}'

def ensure_model():
    if not MODEL_READY: raise RuntimeError(f"Model unavailable. Check {MODEL_PATH} and dependencies.")

def run_image(path, threshold, margin, progress=gr.Progress()):
    try:
        ensure_model()
        if not path: raise ValueError('Upload an image first.')
        progress(.15, desc='Checking image')
        meta = image_metadata(path)
        progress(.4, desc='Running model')
        score = manager.predict_image_path(path)
        decision = classify_score(score, float(threshold), float(margin))
        progress(1, desc='Complete')
        md = f"""### {decision['label']}\n{decision['explanation']}\n\n| Measure | Result |\n|---|---:|\n| Fake score | **{score:.2%}** |\n| Real score | **{1-score:.2%}** |\n| Threshold | {float(threshold):.2f} |\n| Uncertainty margin | ±{float(margin):.2f} |\n\n_Model output scores are not calibrated forensic probabilities._"""
        return md, {**meta, 'assessment': decision['label'], 'fake_score': round(score, 6),
                    'real_score': round(1-score, 6), 'threshold': float(threshold),
                    'uncertainty_margin': float(margin), 'model': Path(MODEL_PATH).name,
                    'model_input': '224x224 RGB float32; model applies internal EfficientNet scaling'}
    except Exception as exc:
        return f'### Image analysis failed\n{exc}', {}

def run_video(path, threshold, margin, progress=gr.Progress()):
    try:
        ensure_model()
        if not path: raise ValueError('Upload a video first.')
        result = analyze_video(path, manager.predict_rgb, float(threshold), float(margin), progress)
        md = f"""### {result['assessment']}\n{result['explanation']}\n\n| Measure | Result |\n|---|---:|\n| Average fake score | **{result['average_fake_score']:.2%}** |\n| Average real score | **{result['average_real_score']:.2%}** |\n| Frames above threshold | {result['frames_above_threshold_percent']:.1f}% |\n| Sampled frames | {result['sampled_frames']} |\n| Duration | {result['duration_seconds']:.1f} seconds |\n| Highest-scoring frame | {result['highest_frame_timestamp']} |\n\n**Limitation:** This applies an image classifier to sampled frames. It does not analyze temporal consistency, audio, lip-sync, or motion; it is not a definitive forensic conclusion."""
        chart = pd.DataFrame(result.pop('chart_data'))
        gallery = result.pop('previews')
        result['model'] = Path(MODEL_PATH).name
        os.makedirs('outputs', exist_ok=True)
        report_path = os.path.join('outputs', 'video_analysis_report.json')
        with open(report_path, 'w', encoding='utf-8') as report_file:
            json.dump(result, report_file, indent=2)
        return md, result, chart, gallery, report_path
    except Exception as exc:
        return f'### Video analysis failed\n{exc}', {}, pd.DataFrame(columns=['timestamp','p_fake']), [], None

CSS = """
.gradio-container{max-width:1180px!important;margin:auto!important}body,.gradio-container{background:#0b1220!important}
#hero{background:linear-gradient(125deg,#101d32,#102b3c 60%,#12394a);border:1px solid #24445b;border-radius:20px;padding:26px}
#hero h1{color:#f2f8ff;letter-spacing:-1px}#hero p{color:#b6c9d8}.result{border:1px solid #29445a;background:#101d2d;border-radius:14px;padding:16px;min-height:110px}
footer{display:none!important}
"""
with gr.Blocks(title='Deepfake Detection Studio', theme=Soft(primary_hue='cyan',secondary_hue='blue',neutral_hue='slate'), css=CSS) as demo:
    with gr.Column(elem_id='hero'):
        gr.Markdown('# Deepfake Detection Studio\nAI-assisted image and video screening with your trained EfficientNetB0 classifier.')
        gr.Markdown(f'**Status:** {MODEL_STATUS}')
    gr.Markdown('Adjust the decision boundary, upload a file, and run analysis. Inference runs on CPU.')
    with gr.Row():
        threshold = gr.Slider(.1,.9,value=.5,step=.01,label='Fake decision threshold')
        margin = gr.Slider(0,.2,value=.05,step=.01,label='Inconclusive margin')
    with gr.Tabs():
        with gr.Tab('Image analysis'):
            with gr.Row():
                with gr.Column():
                    image_in = gr.Image(type='filepath', sources=['upload'], label='Image (JPG / PNG / WEBP)', height=330)
                    with gr.Row():
                        image_btn = gr.Button('Analyze image', variant='primary')
                        image_clear = gr.Button('Clear')
                with gr.Column():
                    image_result = gr.Markdown(elem_classes=['result'])
                    image_json = gr.JSON(label='Technical details')
            image_btn.click(run_image,[image_in,threshold,margin],[image_result,image_json])
            image_clear.click(lambda: (None,'',{}),outputs=[image_in,image_result,image_json])
        with gr.Tab('Video analysis'):
            with gr.Row():
                with gr.Column():
                    video_in = gr.Video(sources=['upload'], label='Video (MP4 / MOV / WEBM)', height=300)
                    gr.Markdown('Limits: 100 MB, 60 seconds, approximately 1 sampled frame per second.')
                    with gr.Row():
                        video_btn = gr.Button('Analyze video',variant='primary')
                        video_clear = gr.Button('Clear')
                with gr.Column():
                    video_result = gr.Markdown(elem_classes=['result'])
                    video_json = gr.JSON(label='Technical details')
            video_plot = gr.LinePlot(x='timestamp',y='p_fake',title='Fake score across sampled frames',x_title='Time (seconds)',y_title='P(fake)',y_lim=[0,1],height=260)
            video_gallery = gr.Gallery(label='Selected sampled frames',columns=3,height=230,object_fit='contain')
            video_report = gr.File(label='Download video analysis report (JSON)')
            video_btn.click(run_video,[video_in,threshold,margin],[video_result,video_json,video_plot,video_gallery,video_report])
            video_clear.click(lambda: (None,'',{},pd.DataFrame(columns=['timestamp','p_fake']),[],None),outputs=[video_in,video_result,video_json,video_plot,video_gallery,video_report])
    gr.Markdown('---\n**Model note:** real = 0, fake = 1. Video screening applies this image classifier to sampled frames; it is not a temporal video model.')

if __name__ == '__main__':
    demo.queue(default_concurrency_limit=1, max_size=8).launch(server_name='0.0.0.0', server_port=int(os.getenv('PORT','7860')))
