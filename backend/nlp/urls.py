from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import CitizenReportViewSet

router = DefaultRouter()
router.register(r'reports', CitizenReportViewSet)

urlpatterns = [
    path('', include(router.urls)),
]
