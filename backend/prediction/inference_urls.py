"""
Inference API URL Configuration
"""
from django.urls import path
from .inference_views import (
    CrimePredictionView,
    AudioClassificationView,
    VideoClassificationView,
    TextClassificationView,
    ImageClassificationView,
    ModelStatusView,
    BatchCrimePredictionView
)

urlpatterns = [
    # Model status
    path('status/', ModelStatusView.as_view(), name='model-status'),

    # Crime prediction
    path('crime/predict/', CrimePredictionView.as_view(), name='crime-predict'),
    path('crime/batch/', BatchCrimePredictionView.as_view(), name='crime-batch-predict'),

    # Audio classification
    path('audio/classify/', AudioClassificationView.as_view(), name='audio-classify'),

    # Video classification
    path('video/classify/', VideoClassificationView.as_view(), name='video-classify'),

    # Text/NLP classification
    path('nlp/classify/', TextClassificationView.as_view(), name='nlp-classify'),

    # Vision/Image classification
    path('vision/classify/', ImageClassificationView.as_view(), name='vision-classify'),
]
