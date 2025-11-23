import React from 'react';
import { MapContainer, TileLayer, Circle, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';

interface PredictionMapProps {
  hotZones: any[];
}

const PredictionMap: React.FC<PredictionMapProps> = ({ hotZones }) => {
  const center: [number, number] = [31.0409, 31.3785];

  const getColor = (riskLevel: string) => {
    switch (riskLevel) {
      case 'high': return 'red';
      case 'medium': return 'orange';
      case 'low': return 'yellow';
      default: return 'blue';
    }
  };

  return (
    <MapContainer center={center} zoom={13} style={{ height: '100%', width: '100%' }}>
      <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
      {hotZones.map((zone) => (
        <Circle
          key={zone.id}
          center={[zone.latitude, zone.longitude]}
          radius={zone.radius || 300}
          pathOptions={{
            color: getColor(zone.risk_level),
            fillColor: getColor(zone.risk_level),
            fillOpacity: 0.3,
          }}
        >
          <Popup>
            <div className="p-2">
              <h3 className="font-bold">{zone.name}</h3>
              <p className="text-sm">Risk: {zone.risk_level}</p>
              <p className="text-sm">Incidents: {zone.incident_count}</p>
            </div>
          </Popup>
        </Circle>
      ))}
    </MapContainer>
  );
};

export default PredictionMap;
