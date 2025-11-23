from django.db import models

class CrimePrediction(models.Model):
    CRIME_TYPES = [
        ('theft', 'سرقة'),
        ('assault', 'اعتداء'),
        ('vandalism', 'تخريب'),
        ('burglary', 'سطو'),
        ('robbery', 'نهب'),
        ('traffic', 'مخالفة مرورية'),
        ('public_order', 'إخلال بالأمن العام'),
        ('drugs', 'مخدرات'),
        ('other', 'أخرى'),
    ]

    location = models.CharField(max_length=200, verbose_name="الموقع")
    latitude = models.FloatField(verbose_name="خط العرض")
    longitude = models.FloatField(verbose_name="خط الطول")
    crime_type = models.CharField(max_length=20, choices=CRIME_TYPES, verbose_name="نوع الجريمة")
    probability = models.FloatField(verbose_name="الاحتمالية")
    predicted_date = models.DateField(verbose_name="التاريخ المتوقع")
    time_window = models.CharField(max_length=50, verbose_name="النافذة الزمنية")
    confidence = models.FloatField(verbose_name="مستوى الثقة")
    factors = models.JSONField(blank=True, null=True, verbose_name="العوامل المساهمة")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="تاريخ الإنشاء")

    class Meta:
        verbose_name = "تنبؤ جريمة"
        verbose_name_plural = "التنبؤ بالجرائم"
        ordering = ['-predicted_date', '-probability']

    def __str__(self):
        return f"{self.get_crime_type_display()} - {self.location}"

class HotZone(models.Model):
    RISK_LEVELS = [
        ('low', 'خطر منخفض'),
        ('medium', 'خطر متوسط'),
        ('high', 'خطر عالي'),
        ('critical', 'خطر حرج'),
    ]

    name = models.CharField(max_length=200, verbose_name="اسم المنطقة")
    latitude = models.FloatField(verbose_name="خط العرض")
    longitude = models.FloatField(verbose_name="خط الطول")
    radius = models.FloatField(verbose_name="نصف القطر (متر)")
    risk_level = models.CharField(max_length=20, choices=RISK_LEVELS, verbose_name="مستوى الخطورة")
    incident_count = models.IntegerField(default=0, verbose_name="عدد الحوادث")
    description = models.TextField(blank=True, verbose_name="الوصف")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="تاريخ الإنشاء")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="تاريخ التحديث")

    class Meta:
        verbose_name = "منطقة ساخنة"
        verbose_name_plural = "المناطق الساخنة"
        ordering = ['-risk_level']

    def __str__(self):
        return f"{self.name} - {self.get_risk_level_display()}"
