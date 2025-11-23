# دليل النشر - Deployment Guide

## متطلبات الخادم

### Hardware
- CPU: 4 cores minimum (8+ recommended)
- RAM: 16GB minimum (32GB+ recommended for AI models)
- Storage: 500GB SSD minimum
- GPU: NVIDIA GPU with 8GB+ VRAM (optional but recommended for AI)

### Software
- Ubuntu 22.04 LTS أو أحدث
- Python 3.10+
- Node.js 18+
- PostgreSQL 14+
- Redis 7+
- Nginx

## خطوات التثبيت

### 1. تحديث النظام
```bash
sudo apt update && sudo apt upgrade -y
```

### 2. تثبيت Python و Dependencies
```bash
sudo apt install python3.10 python3-pip python3-venv
sudo apt install postgresql postgresql-contrib
sudo apt install redis-server
sudo apt install nginx
```

### 3. تثبيت Node.js
```bash
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs
```

### 4. إعداد PostgreSQL
```bash
sudo -u postgres psql

CREATE DATABASE smart_city_db;
CREATE USER smart_city_user WITH PASSWORD 'strong_password_here';
ALTER ROLE smart_city_user SET client_encoding TO 'utf8';
ALTER ROLE smart_city_user SET default_transaction_isolation TO 'read committed';
ALTER ROLE smart_city_user SET timezone TO 'Africa/Cairo';
GRANT ALL PRIVILEGES ON DATABASE smart_city_db TO smart_city_user;
\q
```

### 5. استنساخ المشروع
```bash
cd /var/www/
sudo git clone <repository_url> smart-city
cd smart-city
```

### 6. إعداد Backend
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install gunicorn
```

### 7. تكوين ملف .env
```bash
cat > .env << 'ENVFILE'
SECRET_KEY='your-very-long-and-random-secret-key-here'
DEBUG=False
ALLOWED_HOSTS=your-domain.com,www.your-domain.com
DATABASE_URL=postgresql://smart_city_user:strong_password_here@localhost:5432/smart_city_db
REDIS_URL=redis://localhost:6379/0
ENVFILE
```

### 8. تطبيق Migrations
```bash
python manage.py migrate
python manage.py collectstatic --noinput
python manage.py createsuperuser
```

### 9. إعداد Frontend
```bash
cd ../frontend
npm install
npm run build
```

### 10. إعداد Gunicorn Service
```bash
sudo cat > /etc/systemd/system/smartcity-backend.service << 'EOF'
[Unit]
Description=Smart City Backend
After=network.target

[Service]
User=www-data
Group=www-data
WorkingDirectory=/var/www/smart-city/backend
Environment="PATH=/var/www/smart-city/backend/venv/bin"
ExecStart=/var/www/smart-city/backend/venv/bin/gunicorn \
          --workers 4 \
          --bind unix:/var/www/smart-city/backend/gunicorn.sock \
          config.wsgi:application

[Install]
WantedBy=multi-user.target
