from rest_framework import serializers
from .models import CitizenReport

class CitizenReportSerializer(serializers.ModelSerializer):
    report_type_display = serializers.CharField(source='get_report_type_display', read_only=True)
    urgency_level_display = serializers.CharField(source='get_urgency_level_display', read_only=True)
    
    class Meta:
        model = CitizenReport
        fields = '__all__'
        read_only_fields = ['created_at', 'processed_by', 'is_processed']

class CitizenReportCreateSerializer(serializers.ModelSerializer):
    class Meta:
        model = CitizenReport
        fields = ['report_text', 'report_type', 'location', 'latitude', 'longitude']
