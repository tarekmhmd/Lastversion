from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import AudioDetectionViewSet

router = DefaultRouter()
router.register(r'detections', AudioDetectionViewSet)

urlpatterns = [
    path('', include(router.urls)),
]
