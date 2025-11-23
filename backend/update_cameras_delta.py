#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cloud Script: Update demo cameras to Delta University, Mansoura
"""
import os
import sys
import django

# Setup Django environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from iot.models import Camera

def update_demo_cameras():
    # إحداثيات جامعة الدلتا
    delta_coords = [
        {'name': 'Camera1', 'latitude': 31.0363, 'longitude': 31.3805},
        {'name': 'Camera2', 'latitude': 31.0363, 'longitude': 31.3820},
    ]

    for cam in delta_coords:
        camera_obj, created = Camera.objects.get_or_create(name=cam['name'])
        camera_obj.latitude = cam['latitude']
        camera_obj.longitude = cam['longitude']
        camera_obj.location = 'Delta University, Mansoura'
        camera_obj.is_active = True
        camera_obj.save()
        if created:
            print(f"[OK] Created {cam['name']} at Delta University")
        else:
            print(f"[OK] Updated {cam['name']} to Delta University")

if __name__ == "__main__":
    update_demo_cameras()
    print("[SUCCESS] All demo cameras updated successfully!")
