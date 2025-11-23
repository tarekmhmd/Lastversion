from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django.http import StreamingHttpResponse
from .models import Camera
from .serializers import CameraSerializer
import cv2
import time

class CameraViewSet(viewsets.ModelViewSet):
    """ViewSet for managing cameras"""
    queryset = Camera.objects.all()
    serializer_class = CameraSerializer

    @action(detail=False, methods=['get'], url_path='active')
    def active_cameras(self, request):
        """Get all active cameras"""
        active = Camera.objects.filter(is_active=True)
        serializer = self.get_serializer(active, many=True)
        return Response(serializer.data)


def generate_camera_feed(camera_name):
    """Generate video stream for a camera"""
    try:
        camera = Camera.objects.get(name=camera_name)
        if not camera.is_active:
            return

        # Try to open the camera stream
        if camera.stream_url:
            cap = cv2.VideoCapture(camera.stream_url)
        else:
            # Default to webcam if no URL specified
            camera_index = int(camera_name.replace('Camera', '')) - 1 if 'Camera' in camera_name else 0
            cap = cv2.VideoCapture(camera_index)

        while True:
            success, frame = cap.read()
            if not success:
                break

            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue

            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            time.sleep(0.033)  # ~30 FPS

        cap.release()
    except Camera.DoesNotExist:
        pass
    except Exception as e:
        print(f"Error generating feed for {camera_name}: {e}")


def camera_feed_view(request, camera_name):
    """View to stream camera feed"""
    return StreamingHttpResponse(
        generate_camera_feed(camera_name),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )
