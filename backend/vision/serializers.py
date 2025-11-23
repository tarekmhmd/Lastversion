from rest_framework import serializers
from .models import Camera, IncidentDetection

class CameraSerializer(serializers.ModelSerializer):
    created_by_name = serializers.CharField(source='created_by.username', read_only=True)
    
    class Meta:
        model = Camera
        fields = '__all__'
        read_only_fields = ['created_by', 'created_at', 'updated_at']

class IncidentDetectionSerializer(serializers.ModelSerializer):
    camera_name = serializers.CharField(source='camera.name', read_only=True)
    camera_location = serializers.CharField(source='camera.location', read_only=True)
    incident_type_display = serializers.CharField(source='get_incident_type_display', read_only=True)
    
    class Meta:
        model = IncidentDetection
        fields = '__all__'
        read_only_fields = ['created_at', 'verified_by', 'verified_at']

class IncidentDetectionListSerializer(serializers.ModelSerializer):
    camera_name = serializers.CharField(source='camera.name', read_only=True)
    
    class Meta:
        model = IncidentDetection
        fields = ['id', 'camera', 'camera_name', 'incident_type', 'confidence', 'priority', 'timestamp', 'is_verified']
