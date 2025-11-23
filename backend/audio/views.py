from rest_framework import viewsets, permissions, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django.utils import timezone
from .models import AudioDetection
from .serializers import AudioDetectionSerializer, AudioDetectionListSerializer

class AudioDetectionViewSet(viewsets.ModelViewSet):
    queryset = AudioDetection.objects.all()
    serializer_class = AudioDetectionSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_serializer_class(self):
        if self.action == 'list':
            return AudioDetectionListSerializer
        return AudioDetectionSerializer
    
    @action(detail=True, methods=['post'])
    def verify(self, request, pk=None):
        detection = self.get_object()
        detection.is_verified = True
        detection.verified_by = request.user
        detection.save()
        serializer = self.get_serializer(detection)
        return Response(serializer.data)
    
    @action(detail=False, methods=['get'])
    def recent(self, request):
        recent = self.queryset.filter(
            timestamp__gte=timezone.now() - timezone.timedelta(hours=24)
        )
        serializer = self.get_serializer(recent, many=True)
        return Response(serializer.data)
    
    @action(detail=False, methods=['get'])
    def stats(self, request):
        stats = {
            'total': self.queryset.count(),
            'verified': self.queryset.filter(is_verified=True).count(),
            'by_type': {},
        }
        for choice in AudioDetection.SOUND_TYPES:
            count = self.queryset.filter(sound_type=choice[0]).count()
            stats['by_type'][choice[0]] = count
        return Response(stats)
