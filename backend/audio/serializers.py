from rest_framework import serializers
from .models import AudioDetection

class AudioDetectionSerializer(serializers.ModelSerializer):
    sound_type_display = serializers.CharField(source='get_sound_type_display', read_only=True)
    priority_display = serializers.CharField(source='get_priority_display', read_only=True)
    
    class Meta:
        model = AudioDetection
        fields = '__all__'
        read_only_fields = ['created_at', 'verified_by']

class AudioDetectionListSerializer(serializers.ModelSerializer):
    class Meta:
        model = AudioDetection
        fields = ['id', 'location', 'sound_type', 'confidence', 'priority', 'timestamp', 'is_verified']
