from django.contrib import admin
from .models import Alert

@admin.register(Alert)
class AlertAdmin(admin.ModelAdmin):
    list_display = ('camera_name', 'alert_type', 'severity', 'confidence', 'is_resolved', 'created_at')
    list_filter = ('alert_type', 'severity', 'is_resolved', 'created_at')
    search_fields = ('camera_name', 'message')
    readonly_fields = ('created_at', 'resolved_at')
    actions = ['mark_as_resolved']

    fieldsets = (
        ('Alert Information', {
            'fields': ('camera_name', 'alert_type', 'severity', 'message', 'confidence')
        }),
        ('Location', {
            'fields': ('latitude', 'longitude')
        }),
        ('Media', {
            'fields': ('image_url',)
        }),
        ('Status', {
            'fields': ('is_resolved', 'created_at', 'resolved_at')
        }),
    )

    def mark_as_resolved(self, request, queryset):
        from django.utils import timezone
        updated = queryset.update(is_resolved=True, resolved_at=timezone.now())
        self.message_user(request, f'{updated} alert(s) marked as resolved.')
    mark_as_resolved.short_description = 'Mark selected alerts as resolved'
