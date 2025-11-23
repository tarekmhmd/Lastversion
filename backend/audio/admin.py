from django.contrib import admin
from .models import AudioDetection

@admin.register(AudioDetection)
class AudioDetectionAdmin(admin.ModelAdmin):
    list_display = ['id', 'location', 'sound_type', 'confidence', 'priority', 'timestamp', 'is_verified']
    list_filter = ['sound_type', 'priority', 'is_verified', 'timestamp']
    search_fields = ['location']
    readonly_fields = ['created_at']
    date_hierarchy = 'timestamp'
    
    fieldsets = (
        ('معلومات الموقع', {
            'fields': ('location', 'latitude', 'longitude')
        }),
        ('تفاصيل الكشف الصوتي', {
            'fields': ('sound_type', 'confidence', 'priority', 'timestamp', 'audio_file')
        }),
        ('القياسات', {
            'fields': ('duration', 'decibel_level')
        }),
        ('التحقق', {
            'fields': ('is_verified', 'verified_by', 'created_at')
        }),
    )
