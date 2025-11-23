from rest_framework import viewsets, permissions, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django.utils import timezone
from django.db.models import Count, Avg, Q
from datetime import timedelta
from .serializers import (
    AnalyticsOverviewSerializer,
    IncidentTrendsSerializer,
    HeatmapDataSerializer,
    PerformanceMetricsSerializer
)

class AnalyticsViewSet(viewsets.ViewSet):
    """
    ViewSet for analytics data aggregation
    """
    permission_classes = [permissions.IsAuthenticated]

    @action(detail=False, methods=['get'])
    def overview(self, request):
        """Get comprehensive analytics overview"""
        from vision.models import Camera, IncidentDetection
        from audio.models import AudioDetection
        from nlp.models import CitizenReport

        now = timezone.now()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        # Camera statistics
        total_cameras = Camera.objects.count()
        online_cameras = Camera.objects.filter(status='online', is_active=True).count()

        # Incident statistics
        total_incidents = IncidentDetection.objects.count()
        incidents_today = IncidentDetection.objects.filter(
            timestamp__gte=today_start
        ).count()

        # Audio detection statistics
        try:
            total_audio_detections = AudioDetection.objects.count()
            critical_audio = AudioDetection.objects.filter(
                priority='critical'
            ).count()
        except:
            total_audio_detections = 0
            critical_audio = 0

        # Citizen report statistics
        try:
            total_reports = CitizenReport.objects.count()
            urgent_reports = CitizenReport.objects.filter(
                urgency_level='critical'
            ).count()
        except:
            total_reports = 0
            urgent_reports = 0

        data = {
            'total_cameras': total_cameras,
            'online_cameras': online_cameras,
            'total_incidents': total_incidents,
            'incidents_today': incidents_today,
            'total_audio_detections': total_audio_detections,
            'critical_audio': critical_audio,
            'total_reports': total_reports,
            'urgent_reports': urgent_reports,
            'period_start': today_start,
            'period_end': now,
        }

        serializer = AnalyticsOverviewSerializer(data)
        return Response(serializer.data)

    @action(detail=False, methods=['get'])
    def incident_trends(self, request):
        """Get incident trends over the last 30 days"""
        from vision.models import IncidentDetection

        days = int(request.query_params.get('days', 30))
        start_date = timezone.now() - timedelta(days=days)

        # Group incidents by date
        incidents_by_date = IncidentDetection.objects.filter(
            timestamp__gte=start_date
        ).extra(
            select={'date': 'DATE(timestamp)'}
        ).values('date').annotate(
            count=Count('id')
        ).order_by('date')

        data = [
            {
                'date': item['date'],
                'count': item['count']
            }
            for item in incidents_by_date
        ]

        serializer = IncidentTrendsSerializer(data, many=True)
        return Response(serializer.data)

    @action(detail=False, methods=['get'])
    def incident_by_type(self, request):
        """Get incident distribution by type"""
        from vision.models import IncidentDetection

        incidents_by_type = IncidentDetection.objects.values(
            'incident_type'
        ).annotate(
            count=Count('id')
        ).order_by('-count')

        return Response(incidents_by_type)

    @action(detail=False, methods=['get'])
    def heatmap(self, request):
        """Get geographic heatmap data based on incidents"""
        from vision.models import IncidentDetection, Camera

        # Get incidents grouped by camera location
        camera_incidents = IncidentDetection.objects.values(
            'camera__latitude',
            'camera__longitude',
            'camera__location'
        ).annotate(
            intensity=Count('id')
        ).filter(intensity__gt=0)

        data = [
            {
                'latitude': item['camera__latitude'],
                'longitude': item['camera__longitude'],
                'intensity': item['intensity'],
                'location_name': item['camera__location']
            }
            for item in camera_incidents
        ]

        serializer = HeatmapDataSerializer(data, many=True)
        return Response(serializer.data)

    @action(detail=False, methods=['get'])
    def performance(self, request):
        """Get system performance metrics"""
        from vision.models import Camera

        # Calculate camera uptime
        total_cameras = Camera.objects.count()
        online_cameras = Camera.objects.filter(status='online').count()
        uptime_percentage = (online_cameras / total_cameras * 100) if total_cameras > 0 else 0

        data = [
            {
                'component': 'Vision AI',
                'uptime_percentage': uptime_percentage,
                'avg_response_time': 150.5,  # Mock data - replace with actual metrics
                'total_requests': 1250,
                'error_count': 5
            },
            {
                'component': 'Audio Detection',
                'uptime_percentage': 99.2,
                'avg_response_time': 85.3,
                'total_requests': 850,
                'error_count': 2
            },
            {
                'component': 'NLP Processing',
                'uptime_percentage': 98.8,
                'avg_response_time': 200.1,
                'total_requests': 450,
                'error_count': 3
            },
            {
                'component': 'Prediction Engine',
                'uptime_percentage': 99.5,
                'avg_response_time': 120.0,
                'total_requests': 320,
                'error_count': 1
            }
        ]

        serializer = PerformanceMetricsSerializer(data, many=True)
        return Response(serializer.data)

    @action(detail=False, methods=['get'])
    def camera_status(self, request):
        """Get camera status distribution"""
        from vision.models import Camera

        camera_status = Camera.objects.values('status').annotate(
            count=Count('id')
        )

        return Response(camera_status)

    @action(detail=False, methods=['get'])
    def priority_distribution(self, request):
        """Get incident priority distribution"""
        from vision.models import IncidentDetection

        priority_dist = IncidentDetection.objects.values('priority').annotate(
            count=Count('id')
        ).order_by('-count')

        return Response(priority_dist)
