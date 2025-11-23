from django.db import models
from django.contrib.auth import get_user_model
from django.utils import timezone

User = get_user_model()

class AudioDetection(models.Model):
    SOUND_TYPES = [
        ('scream', 'صراخ'),
        ('gunshot', 'طلقة نار'),
        ('explosion', 'انفجار'),
        ('crash', 'اصطدام/حادث'),
        ('siren', 'صافرة إنذار'),
        ('glass_break', 'كسر زجاج'),
        ('alarm', 'صوت إنذار'),
        ('fighting', 'شجار'),
        ('other', 'أخرى'),
    ]

    PRIORITY_LEVELS = [
        ('low', 'منخفض'),
        ('medium', 'متوسط'),
        ('high', 'عالي'),
        ('critical', 'حرج'),
    ]

    location = models.CharField(max_length=200, verbose_name="الموقع")
    latitude = models.FloatField(verbose_name="خط العرض")
    longitude = models.FloatField(verbose_name="خط الطول")
    sound_type = models.CharField(max_length=20, choices=SOUND_TYPES, verbose_name="نوع الصوت")
    confidence = models.FloatField(verbose_name="مستوى الثقة")
    priority = models.CharField(max_length=20, choices=PRIORITY_LEVELS, default='medium', verbose_name="الأولوية")
    timestamp = models.DateTimeField(default=timezone.now, verbose_name="الوقت")
    audio_file = models.FileField(upload_to='audio_detections/%Y/%m/%d/', verbose_name="ملف الصوت")
    duration = models.FloatField(default=0, verbose_name="المدة (بالثواني)")
    decibel_level = models.FloatField(blank=True, null=True, verbose_name="مستوى الديسيبل")
    is_verified = models.BooleanField(default=False, verbose_name="تم التحقق")
    verified_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, verbose_name="تم التحقق بواسطة")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="تاريخ الإنشاء")

    class Meta:
        verbose_name = "كشف صوتي"
        verbose_name_plural = "الكشف الصوتي"
        ordering = ['-timestamp']

    def __str__(self):
        return f"{self.get_sound_type_display()} - {self.location}"
