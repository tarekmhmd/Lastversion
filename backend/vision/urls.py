from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import CameraViewSet, IncidentDetectionViewSet

router = DefaultRouter()
router.register(r'cameras', CameraViewSet)
router.register(r'incidents', IncidentDetectionViewSet)

urlpatterns = [
    path('', include(router.urls)),
]
