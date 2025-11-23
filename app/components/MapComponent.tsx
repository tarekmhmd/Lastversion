'use client';

import React, { useEffect, useRef } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { Location, MapProps } from '../types';

// إصلاح أيقونات Leaflet الافتراضية
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

export default function MapComponent({ locations, selectedLocation, onLocationSelect }: MapProps) {
  const mapRef = useRef<L.Map | null>(null);
  const markersRef = useRef<L.Marker[]>([]);

  useEffect(() => {
    // تهيئة الخريطة
    if (!mapRef.current) {
      mapRef.current = L.map('map').setView([24.7136, 46.6753], 12);
      
      L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors'
      }).addTo(mapRef.current);
    }

    return () => {
      if (mapRef.current) {
        mapRef.current.remove();
        mapRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    // إزالة العلامات القديمة
    markersRef.current.forEach(marker => {
      if (mapRef.current) {
        mapRef.current.removeLayer(marker);
      }
    });
    markersRef.current = [];

    // إضافة علامات جديدة
    if (mapRef.current) {
      locations.forEach(location => {
        const position: [number, number] = location.position || [location.lat, location.lng];
        const marker = L.marker(position)
          .addTo(mapRef.current!)
          .bindPopup(`
            <div class="text-right">
              <h3 class="font-bold">${location.name}</h3>
              <p>النوع: ${location.type || 'غير محدد'}</p>
              <p>البلد: ${location.country}</p>
              ${location.timestamp ? `<p>التاريخ: ${location.timestamp}</p>` : ''}
            </div>
          `);
        
        marker.on('click', () => {
          onLocationSelect(location);
        });

        markersRef.current.push(marker);
      });
    }
  }, [locations, selectedLocation, onLocationSelect]);

  return <div id="map" className="w-full h-full" />;
}