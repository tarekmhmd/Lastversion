from rest_framework import serializers

class AnalyticsOverviewSerializer(serializers.Serializer):
    """Serializer for overall analytics overview"""
    total_cameras = serializers.IntegerField()
    online_cameras = serializers.IntegerField()
    total_incidents = serializers.IntegerField()
    incidents_today = serializers.IntegerField()
    total_audio_detections = serializers.IntegerField()
    critical_audio = serializers.IntegerField()
    total_reports = serializers.IntegerField()
    urgent_reports = serializers.IntegerField()
    period_start = serializers.DateTimeField()
    period_end = serializers.DateTimeField()

class IncidentTrendsSerializer(serializers.Serializer):
    """Serializer for incident trends over time"""
    date = serializers.DateField()
    count = serializers.IntegerField()
    incident_type = serializers.CharField(required=False)

class HeatmapDataSerializer(serializers.Serializer):
    """Serializer for geographic heatmap data"""
    latitude = serializers.FloatField()
    longitude = serializers.FloatField()
    intensity = serializers.IntegerField()
    location_name = serializers.CharField()

class PerformanceMetricsSerializer(serializers.Serializer):
    """Serializer for system performance metrics"""
    component = serializers.CharField()
    uptime_percentage = serializers.FloatField()
    avg_response_time = serializers.FloatField()
    total_requests = serializers.IntegerField()
    error_count = serializers.IntegerField()
