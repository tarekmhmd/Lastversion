#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cloud Script: Generate detailed AI project report for Smart AI City
Output: wholecode.md
"""
import os
import sys
import django

# Setup Django environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

def generate_ai_report(output_file="wholecode.md"):
    lines = []

    lines.append("# Smart AI City - Detailed AI Project Report\n")
    lines.append("Generated automatically by Cloud Script\n")
    lines.append("="*80 + "\n")

    # 1. Vision / Object Detection
    lines.append("## 1️⃣ Vision / Object Detection (YOLOv8)\n")
    lines.append("- **Model Path:** `models/vision/yolov8n.pt`\n")
    lines.append("- **Purpose:** Detect weapons, fire, explosions, crowds, suspicious activity.\n")
    lines.append("- **Data Needed:** Annotated images or video frames with bounding boxes for objects.\n")
    lines.append("- **Training Instructions:**\n")
    lines.append("  1. Collect video or image data from cameras.\n")
    lines.append("  2. Annotate objects using tools like LabelImg or Roboflow.\n")
    lines.append("  3. Train model using:\n")
    lines.append("     ```\n     yolo train data=data.yaml model=yolov8n.pt\n     ```\n")
    lines.append("- **Output:** Alerts created in Django database when incidents are detected.\n")

    # 2. Audio Detection (if any)
    lines.append("## 2️⃣ Audio / Sound Detection\n")
    lines.append("- **Model Path:** `models/audio/*.pt` (if exists)\n")
    lines.append("- **Purpose:** Detect alarms, gunshots, screams, etc.\n")
    lines.append("- **Data Needed:** Audio clips labeled by event type.\n")
    lines.append("- **Training Instructions:** Use PyTorch/torchaudio to fine-tune on labeled audio dataset.\n")

    # 3. NLP / Text Analysis
    lines.append("## 3️⃣ NLP / Text Analysis (Alerts / Logs)\n")
    lines.append("- **Purpose:** Classify alert messages, generate summaries, detect patterns.\n")
    lines.append("- **Data Needed:** Alert logs, textual messages, labeled categories.\n")
    lines.append("- **Training Instructions:** Use HuggingFace Transformers, fine-tune on log dataset.\n")

    # 4. IOT / Camera Streaming
    lines.append("## 4️⃣ IOT / Camera Stream Management\n")
    lines.append("- **Purpose:** Stream multiple cameras, preprocess frames, feed AI models.\n")
    lines.append("- **Data Needed:** Live video streams or demo videos.\n")
    lines.append("- **Implementation:** `CameraProcessor` class reads frames and runs AI detection.\n")
    lines.append("- **Demo Cameras:** Defined in `MultiCameraManager.create_demo_cameras()`.\n")

    # 5. Project Workflow
    lines.append("## 5️⃣ Recommended Workflow\n")
    lines.append("1. Collect raw video/audio data from cameras.\n")
    lines.append("2. Annotate video frames and audio clips.\n")
    lines.append("3. Train vision/audio/NLP models.\n")
    lines.append("4. Test models on live streams.\n")
    lines.append("5. Alerts are automatically saved to Django DB.\n")

    # 6. File structure highlights
    lines.append("## 6️⃣ Important Folders & Files\n")
    lines.append("- `backend/iot/models.py` → Camera definitions.\n")
    lines.append("- `backend/alerts/models.py` → Alert definitions.\n")
    lines.append("- `frontend/components/CameraMapDashboard.tsx` → Frontend display of camera streams.\n")
    lines.append("- `models/vision/` → YOLO models.\n")
    lines.append("- `models/audio/` → Audio models (if any).\n")

    # 7. Notes
    lines.append("## 7️⃣ Notes & Recommendations\n")
    lines.append("- Ensure camera streams are active and reachable.\n")
    lines.append("- Maintain consistent naming for cameras and alerts.\n")
    lines.append("- Annotated datasets are key for model accuracy.\n")
    lines.append("- Regularly retrain AI models with new data for better detection.\n")

    # Write to file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[SUCCESS] AI project report generated successfully: {output_file}")

if __name__ == "__main__":
    generate_ai_report()
