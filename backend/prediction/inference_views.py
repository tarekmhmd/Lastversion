"""
SmartAiCity - Inference API Views
=================================
REST API endpoints for model inference
"""

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from django.utils import timezone
from datetime import datetime
import numpy as np
import logging

# Import inference utilities
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'utils'))
from model_inference import (
    predict_crime,
    predict_audio,
    predict_video,
    predict_text,
    predict_image
)

logger = logging.getLogger(__name__)


class CrimePredictionView(APIView):
    """
    Crime prediction API endpoint

    POST /api/inference/crime/predict/
    Body: {
        "location": "Downtown Mansoura",
        "latitude": 31.0409,
        "longitude": 31.3785,
        "date": "2024-10-27",
        "hour": 18
    }
    """
    permission_classes = [IsAuthenticated]

    def post(self, request):
        try:
            # Extract parameters
            location = request.data.get('location', 'Unknown')
            latitude = float(request.data.get('latitude'))
            longitude = float(request.data.get('longitude'))
            date_str = request.data.get('date')
            hour = int(request.data.get('hour', datetime.now().hour))

            # Parse date
            if date_str:
                date = datetime.strptime(date_str, '%Y-%m-%d')
            else:
                date = datetime.now()

            # Get prediction
            result = predict_crime(location, latitude, longitude, date, hour)

            if result['success']:
                return Response(result, status=status.HTTP_200_OK)
            else:
                return Response(
                    {'error': result['error']},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

        except ValueError as e:
            return Response(
                {'error': f'Invalid input parameters: {str(e)}'},
                status=status.HTTP_400_BAD_REQUEST
            )
        except Exception as e:
            logger.error(f"Crime prediction error: {str(e)}")
            return Response(
                {'error': 'Internal server error'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class AudioClassificationView(APIView):
    """
    Audio classification API endpoint

    POST /api/inference/audio/classify/
    Body: {
        "features": [array of audio features]
    }
    """
    permission_classes = [IsAuthenticated]

    def post(self, request):
        try:
            features = request.data.get('features')

            if not features:
                return Response(
                    {'error': 'Audio features are required'},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Convert to numpy array
            features_array = np.array(features, dtype=np.float32)

            # Get prediction
            result = predict_audio(features_array)

            if result['success']:
                return Response(result, status=status.HTTP_200_OK)
            else:
                return Response(
                    {'error': result['error']},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

        except Exception as e:
            logger.error(f"Audio classification error: {str(e)}")
            return Response(
                {'error': 'Internal server error'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class VideoClassificationView(APIView):
    """
    Video classification API endpoint

    POST /api/inference/video/classify/
    Body: {
        "frames": [array of video frames]
    }
    """
    permission_classes = [IsAuthenticated]

    def post(self, request):
        try:
            frames = request.data.get('frames')

            if not frames:
                return Response(
                    {'error': 'Video frames are required'},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Convert to numpy array
            frames_array = np.array(frames, dtype=np.float32)

            # Get prediction
            result = predict_video(frames_array)

            if result['success']:
                return Response(result, status=status.HTTP_200_OK)
            else:
                return Response(
                    {'error': result['error']},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

        except Exception as e:
            logger.error(f"Video classification error: {str(e)}")
            return Response(
                {'error': 'Internal server error'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class TextClassificationView(APIView):
    """
    Text/NLP classification API endpoint

    POST /api/inference/nlp/classify/
    Body: {
        "text": "Your text here"
    }
    """
    permission_classes = [IsAuthenticated]

    def post(self, request):
        try:
            text = request.data.get('text', '')

            if not text:
                return Response(
                    {'error': 'Text is required'},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Get prediction
            result = predict_text(text)

            if result['success']:
                return Response(result, status=status.HTTP_200_OK)
            else:
                return Response(
                    {'error': result['error']},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

        except Exception as e:
            logger.error(f"Text classification error: {str(e)}")
            return Response(
                {'error': 'Internal server error'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class ImageClassificationView(APIView):
    """
    Image classification API endpoint

    POST /api/inference/vision/classify/
    Body: {
        "image": [array representing image]
    }
    """
    permission_classes = [IsAuthenticated]

    def post(self, request):
        try:
            image = request.data.get('image')

            if not image:
                return Response(
                    {'error': 'Image data is required'},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Convert to numpy array
            image_array = np.array(image, dtype=np.float32)

            # Get prediction
            result = predict_image(image_array)

            if result['success']:
                return Response(result, status=status.HTTP_200_OK)
            else:
                return Response(
                    {'error': result['error']},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

        except Exception as e:
            logger.error(f"Image classification error: {str(e)}")
            return Response(
                {'error': 'Internal server error'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class ModelStatusView(APIView):
    """
    Check if models are loaded and ready

    GET /api/inference/status/
    """
    permission_classes = [IsAuthenticated]

    def get(self, request):
        from model_inference import _MODELS, MODEL_DIR

        status_data = {
            'models_loaded': list(_MODELS.keys()),
            'model_directory': str(MODEL_DIR),
            'available_models': []
        }

        # Check which model files exist
        model_files = {
            'crime': 'crime_model.pkl',
            'audio': 'audio_model.pth',
            'video': 'video_model.pth',
            'nlp': 'nlp_model.pth',
            'vision': 'vision_model.pth'
        }

        for model_name, file_name in model_files.items():
            model_path = MODEL_DIR / file_name
            if model_path.exists():
                status_data['available_models'].append(model_name)

        return Response(status_data, status=status.HTTP_200_OK)


class BatchCrimePredictionView(APIView):
    """
    Batch crime prediction for multiple locations

    POST /api/inference/crime/batch/
    Body: {
        "predictions": [
            {"location": "...", "latitude": ..., "longitude": ..., "date": "...", "hour": ...},
            ...
        ]
    }
    """
    permission_classes = [IsAuthenticated]

    def post(self, request):
        try:
            predictions_input = request.data.get('predictions', [])

            if not predictions_input:
                return Response(
                    {'error': 'Predictions array is required'},
                    status=status.HTTP_400_BAD_REQUEST
                )

            results = []

            for pred_data in predictions_input:
                location = pred_data.get('location', 'Unknown')
                latitude = float(pred_data.get('latitude'))
                longitude = float(pred_data.get('longitude'))
                date_str = pred_data.get('date')
                hour = int(pred_data.get('hour', datetime.now().hour))

                # Parse date
                if date_str:
                    date = datetime.strptime(date_str, '%Y-%m-%d')
                else:
                    date = datetime.now()

                # Get prediction
                result = predict_crime(location, latitude, longitude, date, hour)
                results.append(result)

            return Response({
                'success': True,
                'predictions': results,
                'total': len(results)
            }, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Batch crime prediction error: {str(e)}")
            return Response(
                {'error': 'Internal server error'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
