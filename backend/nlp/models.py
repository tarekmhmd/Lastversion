from django.db import models
from django.contrib.auth import get_user_model

User = get_user_model()

class CitizenReport(models.Model):
    REPORT_TYPES = [
        ('crime', 'بلاغ جريمة'),
        ('accident', 'بلاغ حادث'),
        ('complaint', 'شكوى'),
        ('suggestion', 'اقتراح'),
        ('emergency', 'طوارئ'),
        ('traffic', 'مرور'),
        ('other', 'أخرى'),
    ]

    URGENCY_LEVELS = [
        ('low', 'منخفض'),
        ('medium', 'متوسط'),
        ('high', 'عالي'),
        ('critical', 'حرج'),
    ]

    report_text = models.TextField(verbose_name="نص البلاغ")
    report_type = models.CharField(max_length=20, choices=REPORT_TYPES, verbose_name="نوع البلاغ")
    location = models.CharField(max_length=200, blank=True, verbose_name="الموقع")
    latitude = models.FloatField(null=True, blank=True, verbose_name="خط العرض")
    longitude = models.FloatField(null=True, blank=True, verbose_name="خط الطول")
    urgency_level = models.CharField(max_length=20, choices=URGENCY_LEVELS, default='medium', verbose_name="مستوى الاستعجال")
    confidence = models.FloatField(default=0.0, verbose_name="مستوى الثقة")
    processed_text = models.TextField(blank=True, verbose_name="النص المعالج")
    entities = models.JSONField(blank=True, null=True, verbose_name="الكيانات المستخرجة")
    is_processed = models.BooleanField(default=False, verbose_name="تم المعالجة")
    processed_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, verbose_name="تم المعالجة بواسطة")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="تاريخ الإنشاء")

    class Meta:
        verbose_name = "بلاغ مواطن"
        verbose_name_plural = "بلاغات المواطنين"
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.get_report_type_display()} - {self.location if self.location else 'موقع غير محدد'}"
