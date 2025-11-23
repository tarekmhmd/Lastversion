from rest_framework import serializers
from .models import CrimePrediction, HotZone

class CrimePredictionSerializer(serializers.ModelSerializer):
    crime_type_display = serializers.CharField(source='get_crime_type_display', read_only=True)
    
    class Meta:
        model = CrimePrediction
        fields = '__all__'
        read_only_fields = ['created_at']

class HotZoneSerializer(serializers.ModelSerializer):
    risk_level_display = serializers.CharField(source='get_risk_level_display', read_only=True)
    
    class Meta:
        model = HotZone
        fields = '__all__'
        read_only_fields = ['created_at', 'updated_at']
