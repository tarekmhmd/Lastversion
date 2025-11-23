from django.contrib import admin
from .models import CrimePrediction, HotZone

@admin.register(CrimePrediction)
class CrimePredictionAdmin(admin.ModelAdmin):
    list_display = ['location', 'crime_type', 'probability', 'predicted_date', 'confidence', 'created_at']
    list_filter = ['crime_type', 'predicted_date', 'created_at']
    search_fields = ['location']
    date_hierarchy = 'predicted_date'
    
    fieldsets = (
        ('معلومات الموقع', {
            'fields': ('location', 'latitude', 'longitude')
        }),
        ('تفاصيل التنبؤ', {
            'fields': ('crime_type', 'probability', 'predicted_date', 'time_window', 'confidence')
        }),
        ('العوامل', {
            'fields': ('factors',)
        }),
    )

@admin.register(HotZone)
class HotZoneAdmin(admin.ModelAdmin):
    list_display = ['name', 'risk_level', 'incident_count', 'created_at', 'updated_at']
    list_filter = ['risk_level', 'created_at']
    search_fields = ['name', 'description']
    
    fieldsets = (
        ('معلومات المنطقة', {
            'fields': ('name', 'latitude', 'longitude', 'radius')
        }),
        ('تقييم المخاطر', {
            'fields': ('risk_level', 'incident_count', 'description')
        }),
        ('التواريخ', {
            'fields': ('created_at', 'updated_at')
        }),
    )
    readonly_fields = ['created_at', 'updated_at']
