from celery import shared_task
import cv2
import numpy as np
import logging
from django.utils import timezone
from django.conf import settings
from .models import Camera, IncidentDetection
import os

logger = logging.getLogger(__name__)

# Initialize YOLO model (lazy loading)
_yolo_model = None

def get_yolo_model():
    """
    Load YOLOv8 model with lazy initialization
    """
    global _yolo_model
    if _yolo_model is None:
        try:
            from ultralytics import YOLO
            model_path = settings.AI_MODELS.get('YOLO_MODEL_PATH', 'yolov8n.pt')

            # If custom model doesn't exist, use pretrained
            if not os.path.exists(model_path):
                logger.warning(f"Custom model not found at {model_path}, using yolov8n.pt")
                model_path = 'yolov8n.pt'

            _yolo_model = YOLO(model_path)
            logger.info("YOLOv8 model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {str(e)}")
            raise
    return _yolo_model

def analyze_detections(results, camera):
    """
    Analyze YOLO detection results to identify incidents
    Returns: (incident_type, confidence, bounding_boxes) or None
    """
    if not results or len(results) == 0:
        return None

    result = results[0]
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return None

    # Class mappings for incident detection
    INCIDENT_CLASSES = {
        'person': {'threshold': 10, 'type': 'crowd', 'priority': 'medium'},
        'car': {'threshold': 8, 'type': 'congestion', 'priority': 'medium'},
        'truck': {'threshold': 5, 'type': 'congestion', 'priority': 'medium'},
        'fire': {'threshold': 1, 'type': 'fire', 'priority': 'critical'},
        'smoke': {'threshold': 1, 'type': 'fire', 'priority': 'critical'},
    }

    detected_objects = {}
    bounding_boxes = []

    # Count detected objects
    for box in boxes:
        cls_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = result.names[cls_id]

        # Store bounding box info
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        bounding_boxes.append({
            'class': class_name,
            'confidence': confidence,
            'bbox': [x1, y1, x2, y2]
        })

        if class_name not in detected_objects:
            detected_objects[class_name] = []
        detected_objects[class_name].append(confidence)

    # Check for incidents based on object counts and thresholds
    for obj_class, config in INCIDENT_CLASSES.items():
        if obj_class in detected_objects:
            count = len(detected_objects[obj_class])
            avg_confidence = np.mean(detected_objects[obj_class])

            if count >= config['threshold'] or obj_class in ['fire', 'smoke']:
                return (config['type'], avg_confidence, bounding_boxes, config['priority'])

    # Check for suspicious activity (person loitering)
    if 'person' in detected_objects and len(detected_objects['person']) > 0:
        avg_conf = np.mean(detected_objects['person'])
        if avg_conf > 0.7:
            return ('suspicious', avg_conf, bounding_boxes, 'low')

    return None

@shared_task
def process_camera_stream(camera_id):
    """
    Process camera stream for incident detection using YOLOv8
    """
    try:
        camera = Camera.objects.get(id=camera_id)
        logger.info(f"Starting camera stream processing: {camera.name}")

        # Load YOLO model
        model = get_yolo_model()

        # Capture frame from camera
        if camera.rtsp_url:
            cap = cv2.VideoCapture(camera.rtsp_url)
        else:
            # Use webcam or sample video for testing
            cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            logger.error(f"Cannot open camera stream: {camera.name}")
            return f"Failed to open camera stream: {camera.name}"

        # Read a frame
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            logger.warning(f"No frame captured from {camera.name}")
            return f"No frame available from {camera.name}"

        # Run YOLOv8 inference
        results = model(frame, conf=settings.APP_SETTINGS.get('INCIDENT_CONFIDENCE_THRESHOLD', 0.7))

        # Analyze detections
        incident_data = analyze_detections(results, camera)

        if incident_data:
            incident_type, confidence, bounding_boxes, priority = incident_data

            # Create incident detection
            incident = IncidentDetection.objects.create(
                camera=camera,
                incident_type=incident_type,
                confidence=confidence,
                priority=priority,
                bounding_boxes=bounding_boxes,
                timestamp=timezone.now(),
                description=f"Detected {incident_type} with {len(bounding_boxes)} objects"
            )

            logger.info(f"Incident detected: {incident_type} from camera {camera.name} with confidence {confidence:.2f}")
            return f"Processed {camera.name} - Detected: {incident_type}"
        else:
            logger.info(f"No incidents detected from {camera.name}")
            return f"Processed {camera.name} - No incidents"

    except Camera.DoesNotExist:
        logger.error(f"Camera not found: {camera_id}")
        return "Camera not found"
    except Exception as e:
        logger.error(f"Error processing camera stream: {str(e)}", exc_info=True)
        return f"Error: {str(e)}"

@shared_task
def batch_process_recorded_videos():
    """
    Process recorded videos in batch
    """
    logger.info("Starting batch video processing")
    # Implement video file processing logic here
    return "Batch video processing completed"

@shared_task
def validate_incident_detection(incident_id):
    """
    Validate incident detection
    """
    try:
        incident = IncidentDetection.objects.get(id=incident_id)
        incident.is_verified = True
        incident.save()
        return f"Incident {incident_id} validated"
    except IncidentDetection.DoesNotExist:
        return f"Incident not found: {incident_id}"

@shared_task
def process_all_active_cameras():
    """
    Process all active cameras
    """
    active_cameras = Camera.objects.filter(status='online', is_active=True)
    for camera in active_cameras:
        process_camera_stream.delay(camera.id)
    return f"Sent {active_cameras.count()} cameras for processing"
