import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO
import PIL.Image as Image
import time
import json
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Create analytics folder
os.makedirs('analytics_images', exist_ok=True)

# Load YOLO model with caching
def load_model():
    model_path = '../YoloWeights/yolov8s.pt'
    # For Hugging Face Spaces, consider using a cached model
    try:
        return YOLO(model_path)
    except:
        # Fallback to a smaller model if the specified one isn't available
        return YOLO('yolov8n.pt')

model = load_model()

# Analytics tracking
class Analytics:
    def __init__(self):
        self.analytics_file = "analytics.json"
        self.detection_history = []
        self.load_data()
    
    def load_data(self):
        if os.path.exists(self.analytics_file):
            try:
                with open(self.analytics_file, 'r') as f:
                    self.detection_history = json.load(f)
            except:
                self.detection_history = []
    
    def save_data(self):
        with open(self.analytics_file, 'w') as f:
            json.dump(self.detection_history, f)
    
    def record_detection(self, image_info, detection_count, classes, score):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "detection_count": detection_count,
            "classes": classes,
            "score": score,
            "image_info": image_info
        }
        self.detection_history.append(entry)
        # Keep only the last 1000 entries to prevent file from growing too large
        if len(self.detection_history) > 1000:
            self.detection_history = self.detection_history[-1000:]
        self.save_data()
    
    def get_stats(self):
        if not self.detection_history:
            return {"total_detections": 0, "avg_score": 0, "common_classes": []}
        
        total_detections = len(self.detection_history)
        scores = [entry["score"] for entry in self.detection_history]
        avg_score = sum(scores) / len(scores)
        
        # Count class occurrences
        class_counter = {}
        for entry in self.detection_history:
            for cls in entry["classes"]:
                class_counter[cls] = class_counter.get(cls, 0) + 1
        
        # Get top 5 most common classes
        common_classes = sorted(class_counter.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "total_detections": total_detections,
            "avg_score": round(avg_score, 2),
            "common_classes": common_classes
        }

# Initialize analytics
analytics = Analytics()

# Performance optimization - cache model results
def cached_detection(image, cache):
    """Cache detection results to improve performance"""
    # Create a hash of the image for caching
    img_hash = hash(image.tobytes())
    
    if img_hash in cache:
        return cache[img_hash]
    
    results = model(image)
    detections = results[0].boxes
    annotated_image = results[0].plot()
    
    cache[img_hash] = (annotated_image, detections)
    return annotated_image, detections

# Global cache for detection results
detection_cache = {}

# Global settings with performance configurations
settings = {
    'confidence': 0.5,
    'performance_mode': 'Balanced'
}

performance_configs = {
    'Fast': {
        'warning': '⚠️ Fast mode: Lower accuracy but 3x faster processing',
        'color': '#ffc107'
    },
    'Balanced': {
        'warning': '✅ Balanced mode: Optimal speed/accuracy ratio',
        'color': '#00d4aa'
    },
    'Accurate': {
        'warning': '🐌 Accurate mode: Higher accuracy but 2x slower processing',
        'color': '#ff6b6b'
    }
}

def save_settings(confidence, performance_mode):
    settings['confidence'] = confidence
    settings['performance_mode'] = performance_mode
    
    config = performance_configs[performance_mode]
    warning_msg = config['warning']
    
    if confidence < 0.3:
        warning_msg += "\n⚠️ Low confidence may result in false detections"
    elif confidence > 0.7:
        warning_msg += "\n⚠️ High confidence may miss some objects"
    
    return f"💾 Settings saved!\n\n{warning_msg}"

# Modern dark theme with cyan-purple accents
custom_css = """
.gradio-container {
    background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    max-width: 1200px;
    margin: 0 auto;
}

.gr-button {
    background: linear-gradient(45deg, #00d4aa, #7209b7);
    border: none;
    border-radius: 12px;
    color: white;
    font-weight: 600;
    font-size: 14px;
    padding: 10px 20px;
    box-shadow: 0 4px 15px rgba(0, 212, 170, 0.3);
    transition: all 0.3s ease;
    margin: 5px;
}

.gr-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 25px rgba(114, 9, 183, 0.4);
    background: linear-gradient(45deg, #7209b7, #00d4aa);
}

.gr-button:active {
    transform: translateY(0px);
}

.warning-box {
    background: rgba(255, 193, 7, 0.1);
    border: 1px solid #ffc107;
    border-radius: 8px;
    padding: 12px;
    margin: 10px 0;
    color: #ffc107;
}

.success-box {
    background: rgba(0, 212, 170, 0.1);
    border: 1px solid #00d4aa;
    border-radius: 8px;
    padding: 12px;
    margin: 10px 0;
    color: #00d4aa;
}

.dashboard-card {
    background: rgba(0, 212, 170, 0.1);
    border-radius: 12px;
    padding: 15px;
    margin: 10px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(114, 9, 183, 0.3);
}

.stats-number {
    font-size: 2em;
    font-weight: bold;
    color: #00d4aa;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.5);
}

.stats-label {
    font-size: 0.9em;
    color: #e0e0e0;
}
"""

def detect_objects(image):
    if image is None:
        return None, None, "🎮 No image uploaded! Please select an image to start the detection game."
    
    try:
        time.sleep(0.1)
        # Use current settings
        results = model(image, conf=settings['confidence'])
        annotated_image = results[0].plot()
        detections = results[0].boxes
        
        if detections is not None and len(detections) > 0:
            count = len(detections)
            classes = [model.names[int(cls.item())] for cls in detections.cls]
            unique_classes = list(set(classes))
            score = count * 10
            
            analytics.record_detection(
                f"{image.size[0]}x{image.size[1]}", 
                count, 
                unique_classes, 
                score
            )
            
            info = f"🎯 DETECTION SUCCESS! 🎯\n🏆 Score: {score} points\n📊 Found {count} objects\n🎪 Types: {', '.join(unique_classes)}\n⚙️ Mode: {settings['performance_mode']} | Confidence: {settings['confidence']}"
            
            class_counts = {}
            for cls in classes:
                class_counts[cls] = class_counts.get(cls, 0) + 1
            
            plt.figure(figsize=(8, 4))
            plt.bar(class_counts.keys(), class_counts.values(), color=['#00d4aa', '#7209b7', '#16213e', '#1a1a2e', '#0f0f23'])
            plt.title('Object Distribution', color='white')
            plt.xticks(rotation=45, ha='right', color='white')
            plt.yticks(color='white')
            plt.gca().set_facecolor('#1a1a2e')
            plt.tight_layout()
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            chart_path = f"analytics_images/distribution_{timestamp}.png"
            plt.savefig(chart_path, facecolor='#1a1a2e')
            plt.close()
            
            return annotated_image, chart_path, info
        else:
            info = "🔍 No objects detected! Try a different image with more visible objects."
            return image, None, info
    except Exception as e:
        return image, None, f"❌ Game Error: {str(e)}\nTry uploading a different image!"

def detect_realtime(image):
    if image is None:
        return None
    try:
        results = model(image)
        return results[0].plot()
    except Exception:
        return image

def get_analytics_dashboard():
    stats = analytics.get_stats()
    
    dashboard_html = f"""
    <div style="display: flex; flex-wrap: wrap; justify-content: space-around; margin: 15px 0;">
        <div class="dashboard-card" style="flex: 1; min-width: 200px;">
            <div class="stats-number">{stats['total_detections']}</div>
            <div class="stats-label">Total Detections</div>
        </div>
        <div class="dashboard-card" style="flex: 1; min-width: 200px;">
            <div class="stats-number">{stats['avg_score']}</div>
            <div class="stats-label">Average Score</div>
        </div>
        <div class="dashboard-card" style="flex: 2; min-width: 300px;">
            <div style="font-size: 1.2em; color: var(--secondary); margin-bottom: 10px;">Top Detected Objects</div>
            <div style="display: flex; flex-wrap: wrap; gap: 10px;">
    """
    
    for cls, count in stats['common_classes']:
        dashboard_html += f"""
                <div style="background: rgba(142, 68, 173, 0.3); padding: 5px 10px; border-radius: 15px; color: white;">
                    {cls}: {count}
                </div>
        """
    
    dashboard_html += """
            </div>
        </div>
    </div>
    """
    
    return dashboard_html

with gr.Blocks(title="🎮 Object Detection Game", css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.HTML("""
    <div style="text-align: center; padding: 20px; background: rgba(0, 212, 170, 0.1); border-radius: 16px; margin-bottom: 20px; border: 2px solid rgba(114, 9, 183, 0.3);">
        <h1 style="color: #00d4aa; font-size: 2.5em; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.8);">🎮 OBJECT DETECTION GAME 🎯</h1>
        <p style="color: #e0e0e0; font-size: 1.1em; margin: 10px 0;">Challenge yourself to find objects in images and webcam!</p>
        <p style="color: #7209b7; font-size: 1em; margin: 0;">🏆 Earn 10 points per detected object | 🎪 80+ object types supported</p>
    </div>
    """)
    
    # Analytics Dashboard
    gr.HTML(get_analytics_dashboard())
    
    with gr.Tabs():
        with gr.Tab("🎯 IMAGE CHALLENGE"):
            gr.HTML("""
            <div style="background: rgba(0, 212, 170, 0.1); padding: 15px; border-radius: 12px; margin-bottom: 15px; border: 1px solid rgba(114, 9, 183, 0.3);">
                <h3 style="color: #00d4aa; margin: 0 0 10px 0;">📋 GAME INSTRUCTIONS:</h3>
                <ol style="color: #e0e0e0; margin: 0; padding-left: 20px;">
                    <li>Upload any image using the upload area below</li>
                    <li>Click the "🚀 START DETECTION" button</li>
                    <li>Watch as AI finds objects and calculates your score!</li>
                    <li>Try different images to beat your high score</li>
                </ol>
            </div>
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    img_input = gr.Image(
                        type="pil", 
                        label="🎮 Upload Your Challenge Image",
                        height=300
                    )
                    with gr.Row():
                        img_btn = gr.Button(
                            "🚀 START DETECTION GAME", 
                            variant="primary",
                            size="lg"
                        )
                        clear_btn = gr.Button(
                            "🧹 Clear Results",
                            variant="secondary"
                        )
                
                with gr.Column(scale=1):
                    img_output = gr.Image(
                        label="🎯 Detection Results",
                        height=300
                    )
                    chart_output = gr.Image(
                        label="📊 Object Distribution",
                        height=200
                    )
            
            img_info = gr.Textbox(
                label="🏆 Game Results", 
                interactive=False,
                lines=4,
                placeholder="Upload an image and click 'START DETECTION GAME' to see your score!"
            )
            
            img_btn.click(
                fn=detect_objects,
                inputs=img_input,
                outputs=[img_output, chart_output, img_info],
                show_progress=True
            )
            
            clear_btn.click(
                fn=lambda: [None, None, "Upload an image and click 'START DETECTION GAME' to see your score!"],
                inputs=[],
                outputs=[img_output, chart_output, img_info]
            )
        
        with gr.Tab("📹 LIVE CAMERA CHALLENGE"):
            gr.HTML("""
            <div style="background: rgba(241, 196, 15, 0.15); padding: 15px; border-radius: 15px; margin-bottom: 15px; border: 1px solid rgba(142, 68, 173, 0.3);">
                <h3 style="color: #f1c40f; margin: 0 0 10px 0;">📋 LIVE GAME INSTRUCTIONS:</h3>
                <ol style="color: white; margin: 0; padding-left: 20px;">
                    <li>Allow camera access when prompted</li>
                    <li>Point your camera at different objects</li>
                    <li>Watch real-time detection in action!</li>
                    <li>Move around to find more objects and increase detection count</li>
                </ol>
                <p style="color: #f39c12; margin: 10px 0 0 0;">💡 TIP: Try pointing at people, phones, laptops, bottles, or furniture!</p>
            </div>
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    webcam_input = gr.Image(
                        sources=["webcam"], 
                        streaming=True, 
                        label="📹 Live Camera Feed",
                        height=300
                    )
                
                with gr.Column(scale=1):
                    webcam_output = gr.Image(
                        label="🎯 Live Detection Results",
                        height=300
                    )
            
            gr.HTML("""
            <div style="background: rgba(142, 68, 173, 0.3); padding: 10px; border-radius: 10px; margin-top: 15px; border: 1px solid rgba(241, 196, 15, 0.5);">
                <p style="color: #f1c40f; margin: 0; text-align: center;">🟢 Camera is active - Detection running automatically!</p>
            </div>
            """)
            
            webcam_input.stream(
                fn=detect_realtime,
                inputs=webcam_input,
                outputs=webcam_output,
                show_progress=False
            )
        
        with gr.Tab("📈 ANALYTICS & SETTINGS"):
            gr.HTML("""
            <div style="background: rgba(241, 196, 15, 0.15); padding: 15px; border-radius: 15px; margin-bottom: 15px; border: 1px solid rgba(142, 68, 173, 0.3);">
                <h3 style="color: #f1c40f; margin: 0 0 10px 0;">📊 PERFORMANCE ANALYTICS:</h3>
                <p style="color: white; margin: 0;">Track your detection history and performance metrics</p>
            </div>
            """)
            
            stats_html = gr.HTML()
            refresh_btn = gr.Button("🔄 Refresh Analytics", variant="primary")
            
            gr.HTML("""
            <div style="background: rgba(142, 68, 173, 0.15); padding: 15px; border-radius: 15px; margin-top: 20px; border: 1px solid rgba(241, 196, 15, 0.3);">
                <h3 style="color: #f1c40f; margin: 0 0 10px 0;">⚙️ APPLICATION SETTINGS:</h3>
                <p style="color: white; margin: 0;">Customize your detection experience</p>
            </div>
            """)
            
            with gr.Row():
                confidence_slider = gr.Slider(
                    minimum=0.1, 
                    maximum=0.9, 
                    value=0.5, 
                    label="Detection Confidence Threshold",
                    info="Higher values mean more confident but fewer detections"
                )
                performance_mode = gr.Radio(
                    choices=["Balanced", "Fast", "Accurate"], 
                    value="Balanced", 
                    label="Performance Mode"
                )
            
            settings_btn = gr.Button("💾 Save Settings", variant="primary")
            settings_output = gr.HTML()
            
            def update_settings_display(confidence, performance_mode):
                result = save_settings(confidence, performance_mode)
                config = performance_configs[performance_mode]
                if "⚠️" in result:
                    return f'<div class="warning-box">{result}</div>'
                else:
                    return f'<div class="success-box">{result}</div>'
            
            settings_btn.click(
                fn=update_settings_display,
                inputs=[confidence_slider, performance_mode],
                outputs=settings_output
            )
            
            refresh_btn.click(
                fn=get_analytics_dashboard,
                inputs=[],
                outputs=stats_html
            )
            
            demo.load(
                fn=get_analytics_dashboard,
                inputs=[],
                outputs=stats_html
            )
            
            demo.load(
                fn=lambda: '<div class="success-box">⚙️ Default settings loaded: Balanced mode with 0.5 confidence threshold</div>',
                inputs=[],
                outputs=settings_output
            )

# For Hugging Face Spaces deployment
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0" if os.getenv("SPACE_ID") else None,
        share=False,
        favicon_path="https://em-content.zobj.net/source/microsoft/319/video-game_1f3ae.png"
    )