from django.contrib import admin
from .models import Camera, IncidentDetection

@admin.register(Camera)
class CameraAdmin(admin.ModelAdmin):
    list_display = ['name', 'location', 'camera_type', 'status', 'is_active', 'created_at']
    list_filter = ['camera_type', 'status', 'is_active', 'created_at']
    search_fields = ['name', 'location', 'ip_address']
    readonly_fields = ['created_at', 'updated_at']
    
    fieldsets = (
        ('معلومات أساسية', {
            'fields': ('name', 'location', 'latitude', 'longitude')
        }),
        ('إعدادات الكاميرا', {
            'fields': ('camera_type', 'status', 'rtsp_url', 'ip_address', 'is_active')
        }),
        ('معلومات إضافية', {
            'fields': ('created_by', 'created_at', 'updated_at')
        }),
    )

@admin.register(IncidentDetection)
class IncidentDetectionAdmin(admin.ModelAdmin):
    list_display = ['id', 'camera', 'incident_type', 'confidence', 'priority', 'timestamp', 'is_verified']
    list_filter = ['incident_type', 'priority', 'is_verified', 'is_false_alarm', 'timestamp']
    search_fields = ['camera__name', 'camera__location', 'description']
    readonly_fields = ['created_at', 'verified_at']
    date_hierarchy = 'timestamp'
    
    fieldsets = (
        ('معلومات الحادث', {
            'fields': ('camera', 'incident_type', 'confidence', 'priority', 'timestamp')
        }),
        ('تفاصيل الكشف', {
            'fields': ('description', 'frame_data', 'bounding_boxes')
        }),
        ('التحقق', {
            'fields': ('is_verified', 'is_false_alarm', 'verified_by', 'verified_at')
        }),
    )
