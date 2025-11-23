from rest_framework import viewsets, permissions, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django.utils import timezone
from .models import Camera, IncidentDetection
from .serializers import CameraSerializer, IncidentDetectionSerializer, IncidentDetectionListSerializer

class CameraViewSet(viewsets.ModelViewSet):
    queryset = Camera.objects.all()
    serializer_class = CameraSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def perform_create(self, serializer):
        serializer.save(created_by=self.request.user)
    
    @action(detail=False, methods=['get'])
    def online(self, request):
        online_cameras = self.queryset.filter(status='online', is_active=True)
        serializer = self.get_serializer(online_cameras, many=True)
        return Response(serializer.data)
    
    @action(detail=False, methods=['get'])
    def stats(self, request):
        stats = {
            'total': self.queryset.count(),
            'online': self.queryset.filter(status='online').count(),
            'offline': self.queryset.filter(status='offline').count(),
            'maintenance': self.queryset.filter(status='maintenance').count(),
        }
        return Response(stats)

class IncidentDetectionViewSet(viewsets.ModelViewSet):
    queryset = IncidentDetection.objects.all()
    serializer_class = IncidentDetectionSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_serializer_class(self):
        if self.action == 'list':
            return IncidentDetectionListSerializer
        return IncidentDetectionSerializer
    
    @action(detail=True, methods=['post'])
    def verify(self, request, pk=None):
        incident = self.get_object()
        incident.is_verified = True
        incident.verified_by = request.user
        incident.verified_at = timezone.now()
        incident.save()
        serializer = self.get_serializer(incident)
        return Response(serializer.data)
    
    @action(detail=False, methods=['get'])
    def recent(self, request):
        recent_incidents = self.queryset.filter(
            timestamp__gte=timezone.now() - timezone.timedelta(hours=24)
        )
        serializer = self.get_serializer(recent_incidents, many=True)
        return Response(serializer.data)
    
    @action(detail=False, methods=['get'])
    def stats(self, request):
        stats = {
            'total': self.queryset.count(),
            'verified': self.queryset.filter(is_verified=True).count(),
            'unverified': self.queryset.filter(is_verified=False).count(),
            'by_type': {},
        }
        for choice in IncidentDetection.INCIDENT_TYPES:
            count = self.queryset.filter(incident_type=choice[0]).count()
            stats['by_type'][choice[0]] = count
        return Response(stats)
