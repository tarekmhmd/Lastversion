from django.contrib import admin
from .models import CitizenReport

@admin.register(CitizenReport)
class CitizenReportAdmin(admin.ModelAdmin):
    list_display = ['id', 'report_type', 'location', 'urgency_level', 'is_processed', 'created_at']
    list_filter = ['report_type', 'urgency_level', 'is_processed', 'created_at']
    search_fields = ['report_text', 'location']
    readonly_fields = ['created_at']
    date_hierarchy = 'created_at'
    
    fieldsets = (
        ('البلاغ', {
            'fields': ('report_text', 'report_type', 'urgency_level')
        }),
        ('الموقع', {
            'fields': ('location', 'latitude', 'longitude')
        }),
        ('المعالجة', {
            'fields': ('is_processed', 'processed_by', 'confidence', 'processed_text', 'entities')
        }),
    )
