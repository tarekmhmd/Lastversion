from django.db import models
from django.contrib.auth import get_user_model
from django.utils import timezone

User = get_user_model()

class Camera(models.Model):
    CAMERA_TYPES = [
        ('cctv', 'CCTV'),
        ('traffic', 'Traffic Camera'),
        ('thermal', 'Thermal Camera'),
        ('dome', 'Dome Camera'),
        ('ptz', 'PTZ Camera'),
    ]

    STATUS_CHOICES = [
        ('online', 'Online'),
        ('offline', 'Offline'),
        ('maintenance', 'Under Maintenance'),
    ]

    name = models.CharField(max_length=200, verbose_name="اسم الكاميرا")
    location = models.CharField(max_length=200, verbose_name="الموقع")
    latitude = models.FloatField(verbose_name="خط العرض")
    longitude = models.FloatField(verbose_name="خط الطول")
    camera_type = models.CharField(max_length=20, choices=CAMERA_TYPES, default='cctv', verbose_name="نوع الكاميرا")
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='online', verbose_name="الحالة")
    rtsp_url = models.URLField(blank=True, null=True, verbose_name="رابط RTSP")
    ip_address = models.GenericIPAddressField(blank=True, null=True, verbose_name="عنوان IP")
    is_active = models.BooleanField(default=True, verbose_name="نشط")
    created_by = models.ForeignKey(User, on_delete=models.CASCADE, verbose_name="تم الإنشاء بواسطة")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="تاريخ الإنشاء")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="تاريخ التحديث")

    class Meta:
        verbose_name = "كاميرا"
        verbose_name_plural = "الكاميرات"
        ordering = ['name']

    def __str__(self):
        return f"{self.name} - {self.location}"

class IncidentDetection(models.Model):
    INCIDENT_TYPES = [
        ('accident', 'حادث مروري'),
        ('congestion', 'ازدحام مروري'),
        ('suspicious', 'نشاط مشبوه'),
        ('violence', 'عنيف/شجار'),
        ('fire', 'حريق'),
        ('crowd', 'تجمع غير طبيعي'),
        ('object', 'جسم مشبوه'),
        ('vandalism', 'تخريب'),
        ('theft', 'سرقة'),
        ('other', 'أخرى'),
    ]

    PRIORITY_LEVELS = [
        ('low', 'منخفض'),
        ('medium', 'متوسط'),
        ('high', 'عالي'),
        ('critical', 'حرج'),
    ]

    camera = models.ForeignKey(Camera, on_delete=models.CASCADE, verbose_name="الكاميرا")
    incident_type = models.CharField(max_length=20, choices=INCIDENT_TYPES, verbose_name="نوع الحادث")
    confidence = models.FloatField(verbose_name="مستوى الثقة")
    priority = models.CharField(max_length=20, choices=PRIORITY_LEVELS, default='medium', verbose_name="الأولوية")
    timestamp = models.DateTimeField(default=timezone.now, verbose_name="الوقت")
    frame_data = models.TextField(blank=True, verbose_name="بيانات الإطار")
    bounding_boxes = models.JSONField(blank=True, null=True, verbose_name="مربعات الاكتشاف")
    description = models.TextField(blank=True, verbose_name="وصف الحادث")
    is_verified = models.BooleanField(default=False, verbose_name="تم التحقق")
    is_false_alarm = models.BooleanField(default=False, verbose_name="إنذار خاطئ")
    verified_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, verbose_name="تم التحقق بواسطة")
    verified_at = models.DateTimeField(null=True, blank=True, verbose_name="وقت التحقق")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="تاريخ الإنشاء")

    class Meta:
        verbose_name = "كشف حادث"
        verbose_name_plural = "كشف الحوادث"
        ordering = ['-timestamp']

    def __str__(self):
        return f"{self.get_incident_type_display()} - {self.camera.name}"
