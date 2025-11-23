#!/usr/bin/env python
"""
Multi-camera streaming with AI processing
Monitors multiple camera feeds and detects incidents using computer vision
"""
import os
import sys
import django
import cv2
import numpy as np
from datetime import datetime
import threading
import time

# Setup Django environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from iot.models import Camera
from alerts.models import Alert

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: YOLO not available. Install ultralytics package for AI detection.")


class CameraProcessor:
    """Process individual camera feed with AI detection"""

    def __init__(self, camera):
        self.camera = camera
        self.running = False
        self.thread = None
        self.model = None

        # Initialize YOLO model if available
        if YOLO_AVAILABLE:
            try:
                model_path = '/mnt/c/SmartAiCity/safe-smart-city-ai/models/vision/yolov8n.pt'
                if os.path.exists(model_path):
                    self.model = YOLO(model_path)
                else:
                    # Use default pretrained model
                    self.model = YOLO('yolov8n.pt')
                print(f"✓ YOLO model loaded for {camera.name}")
            except Exception as e:
                print(f"✗ Failed to load YOLO model: {e}")

        # Define dangerous objects and scenarios
        self.dangerous_classes = {
            'knife', 'gun', 'weapon', 'fire', 'explosion'
        }

        self.suspicious_classes = {
            'person': 5,  # More than 5 people might indicate crowding
        }

    def detect_incidents(self, frame):
        """Detect potential security incidents in frame"""
        if not self.model:
            return []

        incidents = []

        try:
            # Run YOLO detection
            results = self.model(frame, verbose=False)

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get class name and confidence
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = self.model.names[cls_id].lower()

                    # Check for dangerous objects
                    if any(danger in class_name for danger in self.dangerous_classes):
                        incidents.append({
                            'type': 'weapon' if 'gun' in class_name or 'knife' in class_name else 'fire',
                            'severity': 'critical',
                            'message': f"⚠️ {class_name.upper()} detected!",
                            'confidence': conf
                        })

                    # Check for person count (crowd detection)
                    if class_name == 'person' and conf > 0.5:
                        person_count = len([b for b in boxes if self.model.names[int(b.cls[0])] == 'person'])
                        if person_count > self.suspicious_classes.get('person', 5):
                            incidents.append({
                                'type': 'crowd',
                                'severity': 'medium',
                                'message': f"Large crowd detected ({person_count} people)",
                                'confidence': conf
                            })
                            break  # Only report once per frame

                    # Detect violence (multiple people in close proximity with high activity)
                    # This is a simplified heuristic
                    if class_name == 'person':
                        person_boxes = [b for b in boxes if self.model.names[int(b.cls[0])] == 'person']
                        if len(person_boxes) >= 2:
                            # Check for overlapping or very close boxes
                            for i, box1 in enumerate(person_boxes):
                                for box2 in person_boxes[i+1:]:
                                    if self._boxes_close(box1.xyxy[0], box2.xyxy[0]):
                                        incidents.append({
                                            'type': 'suspicious',
                                            'severity': 'high',
                                            'message': "Suspicious activity detected (people in close proximity)",
                                            'confidence': min(float(box1.conf[0]), float(box2.conf[0]))
                                        })
                                        break
                                if incidents:
                                    break
                            if incidents:
                                break

        except Exception as e:
            print(f"Detection error for {self.camera.name}: {e}")

        return incidents

    def _boxes_close(self, box1, box2, threshold=50):
        """Check if two bounding boxes are close to each other"""
        # Calculate center points
        center1 = [(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2]
        center2 = [(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2]

        # Calculate Euclidean distance
        distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        return distance < threshold

    def save_alert(self, incident):
        """Save detected incident as alert"""
        try:
            Alert.objects.create(
                camera_name=self.camera.name,
                alert_type=incident['type'],
                severity=incident['severity'],
                message=incident['message'],
                confidence=incident['confidence'],
                latitude=self.camera.latitude,
                longitude=self.camera.longitude
            )
            print(f"📢 Alert saved: {incident['message']} from {self.camera.name}")
        except Exception as e:
            print(f"Error saving alert: {e}")

    def process_stream(self):
        """Main processing loop for camera stream"""
        print(f"🎥 Starting camera processor for {self.camera.name}")

        # Open video capture
        if self.camera.stream_url:
            cap = cv2.VideoCapture(self.camera.stream_url)
        else:
            # Use webcam index based on camera name
            camera_index = int(self.camera.name.replace('Camera', '')) - 1 if 'Camera' in self.camera.name else 0
            cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            print(f"✗ Failed to open camera {self.camera.name}")
            return

        frame_count = 0
        last_alert_time = {}

        while self.running:
            ret, frame = cap.read()
            if not ret:
                print(f"✗ Failed to read frame from {self.camera.name}")
                time.sleep(1)
                continue

            frame_count += 1

            # Process every 30th frame (~1 per second at 30fps)
            if frame_count % 30 == 0:
                incidents = self.detect_incidents(frame)

                # Save alerts (with throttling to avoid spam)
                current_time = time.time()
                for incident in incidents:
                    incident_key = f"{incident['type']}_{incident['severity']}"

                    # Only save alert if 60 seconds have passed since last similar alert
                    if incident_key not in last_alert_time or \
                       current_time - last_alert_time[incident_key] > 60:
                        self.save_alert(incident)
                        last_alert_time[incident_key] = current_time

            time.sleep(0.033)  # ~30 FPS

        cap.release()
        print(f"🛑 Stopped camera processor for {self.camera.name}")

    def start(self):
        """Start processing in a separate thread"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self.process_stream)
            self.thread.daemon = True
            self.thread.start()

    def stop(self):
        """Stop processing"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)


class MultiCameraManager:
    """Manage multiple camera processors"""

    def __init__(self):
        self.processors = {}

    def start_all_cameras(self):
        """Start processing all active cameras"""
        print("🚀 Starting Multi-Camera AI System")
        print("=" * 60)

        # Get all active cameras from database
        cameras = Camera.objects.filter(is_active=True)

        if not cameras.exists():
            print("⚠️  No active cameras found in database.")
            print("   Creating demo cameras...")
            self.create_demo_cameras()
            cameras = Camera.objects.filter(is_active=True)

        print(f"📹 Found {cameras.count()} active camera(s)")

        # Start processor for each camera
        for camera in cameras:
            processor = CameraProcessor(camera)
            processor.start()
            self.processors[camera.name] = processor
            print(f"✓ Started processor for {camera.name}")

        print("=" * 60)
        print("✅ All cameras are running!")
        print("Press Ctrl+C to stop...")

    def create_demo_cameras(self):
        """Create demo cameras if none exist"""
        demo_cameras = [
            {
                'name': 'Camera1',
                'location': 'Downtown Cairo',
                'latitude': 30.0444,
                'longitude': 31.2357,
                'is_active': True
            },
            {
                'name': 'Camera2',
                'location': 'Nasr City',
                'latitude': 30.0626,
                'longitude': 31.2497,
                'is_active': True
            }
        ]

        for cam_data in demo_cameras:
            Camera.objects.get_or_create(
                name=cam_data['name'],
                defaults=cam_data
            )

        print(f"✓ Created {len(demo_cameras)} demo cameras")

    def stop_all(self):
        """Stop all camera processors"""
        print("\n🛑 Stopping all cameras...")
        for name, processor in self.processors.items():
            processor.stop()
            print(f"✓ Stopped {name}")
        print("👋 Goodbye!")


def main():
    """Main entry point"""
    manager = MultiCameraManager()

    try:
        manager.start_all_cameras()

        # Keep running until interrupted
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n⚠️  Interrupt received...")
        manager.stop_all()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        manager.stop_all()


if __name__ == '__main__':
    main()
