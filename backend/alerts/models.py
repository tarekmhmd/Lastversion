from django.db import models
from django.utils import timezone

class Alert(models.Model):
    """Model to represent security alerts detected by AI"""
    SEVERITY_CHOICES = [
        ('low', 'Low'),
        ('medium', 'Medium'),
        ('high', 'High'),
        ('critical', 'Critical'),
    ]

    ALERT_TYPE_CHOICES = [
        ('weapon', 'Weapon Detection'),
        ('violence', 'Violence'),
        ('crowd', 'Crowd Anomaly'),
        ('fire', 'Fire Detection'),
        ('accident', 'Accident'),
        ('suspicious', 'Suspicious Activity'),
        ('other', 'Other'),
    ]

    camera_name = models.CharField(max_length=100)
    alert_type = models.CharField(max_length=50, choices=ALERT_TYPE_CHOICES, default='other')
    severity = models.CharField(max_length=20, choices=SEVERITY_CHOICES, default='medium')
    message = models.TextField()
    confidence = models.FloatField(default=0.0)
    latitude = models.FloatField(null=True, blank=True)
    longitude = models.FloatField(null=True, blank=True)
    image_url = models.URLField(max_length=500, blank=True, null=True)
    is_resolved = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    resolved_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.alert_type} - {self.camera_name} ({self.created_at})"
