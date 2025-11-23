from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import CrimePredictionViewSet, HotZoneViewSet

router = DefaultRouter()
router.register(r'predictions', CrimePredictionViewSet)
router.register(r'hotzones', HotZoneViewSet)

urlpatterns = [
    path('', include(router.urls)),
]
