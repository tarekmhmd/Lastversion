Smart AI City - دليل التشغيل

هذا المشروع يحتوي على Backend (Django/Python)، Frontend (Next.js/React)، ويمكن تشغيله باستخدام Docker.

---

المتطلبات الأساسية
- Node.js ≥ 18
- Python ≥ 3.11
- Docker و Docker Compose (اختياري)

---

1️⃣ تشغيل الـ Backend (Django)

إعداد البيئة:

cd backend
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
pip install -r requirements.txt

تشغيل السيرفر محلياً:

cp .env.example .env
python manage.py runserver
# السيرفر سيعمل على http://127.0.0.1:8000

---

2️⃣ إنشاء حساب Admin على Django

cd backend
# تفعيل البيئة الافتراضية إذا لم تكن مفعلة
python manage.py createsuperuser

أدخل البيانات المطلوبة:
Username: admin
Email: admin@example.com   # اختياري
Password: ********
Password (again): ********

ثم شغّل السيرفر وادخل على لوحة الإدارة:
http://127.0.0.1:8000/admin

---

3️⃣ تشغيل الـ Frontend (Next.js)

تثبيت الحزم:

cd frontend
npm install

تشغيل التطبيق محلياً:

npm run dev
# التطبيق سيكون متاح على http://localhost:3000

---

4️⃣ تشغيل المشروع باستخدام Docker (اختياري)

بناء وتشغيل الحاويات:

docker-compose up --build

ملاحظات:
- الـ Backend عادة على البورت: 8000
- الـ Frontend عادة على البورت: 3000
- يمكن تعديل البورتات في docker-compose.yml إذا احتجت

---

5️⃣ نصائح

- لتحديث الحزم في Backend: pip install -r requirements.txt --upgrade
- لتحديث حزم Frontend: npm update
- إذا ظهر خطأ في Leaflet أو Geocoder، تأكد من تشغيل Frontend بعد تثبيت الحزم وتفعيل البيئة
