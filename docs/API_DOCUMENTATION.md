# API Documentation - توثيق API

## نظرة عامة
API Documentation للنظام Safe Smart City AI

## Base URL
```
http://localhost:8000/api
```

## Authentication
استخدام JWT Token:
```bash
# الحصول على Token
POST /api/token/
{
  "username": "admin",
  "password": "admin123"
}

# استخدام Token
Headers: {
  "Authorization": "Bearer <access_token>"
}

# تحديث Token
POST /api/token/refresh/
{
  "refresh": "<refresh_token>"
}
```

## Vision API - نظام الرؤية

### 1. الكاميرات

#### قائمة الكاميرات
```bash
GET /api/vision/cameras/
```

Response:
```json
[
  {
    "id": 1,
    "name": "كاميرا وسط البلد",
    "location": "ميدان الشهداء - وسط المنصورة",
    "latitude": 31.0409,
    "longitude": 31.3785,
    "camera_type": "cctv",
    "status": "online",
    "is_active": true
  }
]
```

#### إضافة كاميرا
```bash
POST /api/vision/cameras/
{
  "name": "كاميرا جديدة",
  "location": "الموقع",
  "latitude": 31.04,
  "longitude": 31.37,
  "camera_type": "cctv"
}
```

#### الكاميرات النشطة فقط
```bash
GET /api/vision/cameras/online/
```

#### إحصائيات الكاميرات
```bash
GET /api/vision/cameras/stats/
```

### 2. كشف الحوادث

#### قائمة الحوادث
```bash
GET /api/vision/incidents/
```

Response:
```json
[
  {
    "id": 1,
    "camera": 1,
    "camera_name": "كاميرا وسط البلد",
    "incident_type": "accident",
    "confidence": 0.85,
    "priority": "high",
    "timestamp": "2024-10-26T14:30:00Z",
    "is_verified": false
  }
]
```

#### الحوادث الأخيرة (24 ساعة)
```bash
GET /api/vision/incidents/recent/
```

#### التحقق من حادث
```bash
POST /api/vision/incidents/{id}/verify/
```

#### إحصائيات الحوادث
```bash
GET /api/vision/incidents/stats/
```

Response:
```json
{
  "total": 100,
  "verified": 75,
  "unverified": 25,
  "by_type": {
    "accident": 30,
    "congestion": 40,
    "suspicious": 20,
    "violence": 10
  }
}
```

## Audio API - نظام الصوت

### كشف الأصوات

#### قائمة الكشوفات
```bash
GET /api/audio/detections/
```

Response:
```json
[
  {
    "id": 1,
    "location": "وسط المنصورة",
    "sound_type": "scream",
    "confidence": 0.92,
    "priority": "critical",
    "timestamp": "2024-10-26T14:30:00Z",
    "is_verified": false
  }
]
```

#### الكشوفات الأخيرة
```bash
GET /api/audio/detections/recent/
```

#### التحقق من كشف
```bash
POST /api/audio/detections/{id}/verify/
```

#### إحصائيات
```bash
GET /api/audio/detections/stats/
```

## Prediction API - نظام التنبؤ

### 1. تنبؤات الجرائم

#### قائمة التنبؤات
```bash
GET /api/prediction/predictions/
```

Response:
```json
[
  {
    "id": 1,
    "location": "وسط المنصورة",
    "crime_type": "theft",
    "probability": 0.75,
    "predicted_date": "2024-10-27",
    "time_window": "18:00-22:00",
    "confidence": 0.82
  }
]
```

#### تنبؤات اليوم
```bash
GET /api/prediction/predictions/today/
```

#### التنبؤات عالية الاحتمالية
```bash
GET /api/prediction/predictions/high_probability/
```

### 2. المناطق الساخنة

#### قائمة المناطق
```bash
GET /api/prediction/hotzones/
```

Response:
```json
[
  {
    "id": 1,
    "name": "وسط المنصورة",
    "risk_level": "high",
    "incident_count": 15,
    "latitude": 31.0409,
    "longitude": 31.3785,
    "radius": 500
  }
]
```

#### المناطق الحرجة فقط
```bash
GET /api/prediction/hotzones/critical/
```

## NLP API - معالجة اللغة الطبيعية

### بلاغات المواطنين

#### قائمة البلاغات
```bash
GET /api/nlp/reports/
```

#### إنشاء بلاغ جديد
```bash
POST /api/nlp/reports/
{
  "report_text": "يوجد حادث مروري في شارع الجلاء",
  "report_type": "accident",
  "location": "شارع الجلاء",
  "latitude": 31.0350,
  "longitude": 31.3650
}
```

Response:
```json
{
  "id": 1,
  "report_text": "يوجد حادث مروري في شارع الجلاء",
  "report_type": "accident",
  "urgency_level": "high",
  "confidence": 0.88,
  "is_processed": false,
  "created_at": "2024-10-26T14:30:00Z"
}
```

#### البلاغات غير المعالجة
```bash
GET /api/nlp/reports/unprocessed/
```

#### البلاغات العاجلة
```bash
GET /api/nlp/reports/urgent/
```

## WebSocket API - التحديثات الفورية

### الاتصال
```javascript
const socket = io('ws://localhost:8000');

// استقبال التحديثات الفورية
socket.on('incident_detected', (data) => {
  console.log('حادث جديد:', data);
});

socket.on('audio_alert', (data) => {
  console.log('تنبيه صوتي:', data);
});

socket.on('crime_prediction', (data) => {
  console.log('تنبؤ جديد:', data);
});
```

## أكواد الحالة - Status Codes

- `200 OK` - طلب ناجح
- `201 Created` - تم الإنشاء بنجاح
- `400 Bad Request` - بيانات خاطئة
- `401 Unauthorized` - غير مصرح
- `403 Forbidden` - ممنوع
- `404 Not Found` - غير موجود
- `500 Internal Server Error` - خطأ في الخادم

## معدل الطلبات - Rate Limiting
- **غير محدود** في بيئة التطوير
- **في الإنتاج**: 1000 طلب/ساعة لكل مستخدم

## أمثلة باستخدام cURL

```bash
# الحصول على Token
curl -X POST http://localhost:8000/api/token/ \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin123"}'

# قائمة الكاميرات
curl -X GET http://localhost:8000/api/vision/cameras/ \
  -H "Authorization: Bearer <token>"

# إنشاء بلاغ
curl -X POST http://localhost:8000/api/nlp/reports/ \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{
    "report_text": "بلاغ تجريبي",
    "report_type": "complaint",
    "location": "المنصورة"
  }'
```

## أمثلة باستخدام Python

```python
import requests

BASE_URL = "http://localhost:8000/api"

# الحصول على Token
response = requests.post(f"{BASE_URL}/token/", json={
    "username": "admin",
    "password": "admin123"
})
token = response.json()["access"]

# Headers
headers = {"Authorization": f"Bearer {token}"}

# قائمة الكاميرات
cameras = requests.get(f"{BASE_URL}/vision/cameras/", headers=headers)
print(cameras.json())

# إنشاء بلاغ
report = requests.post(f"{BASE_URL}/nlp/reports/", 
    headers=headers,
    json={
        "report_text": "بلاغ تجريبي",
        "report_type": "complaint",
        "location": "المنصورة"
    }
)
print(report.json())
```

## Support
للدعم والاستفسارات: admin@mansoura-smartcity.eg
