from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import CameraViewSet, camera_feed_view

router = DefaultRouter()
router.register(r'cameras', CameraViewSet, basename='camera')

urlpatterns = [
    path('', include(router.urls)),
    path('camera_feed/<str:camera_name>/', camera_feed_view, name='camera_feed'),
]
