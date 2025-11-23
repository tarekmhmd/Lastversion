export interface Location {
  id: number | string;
  name: string;
  lat: number;
  lng: number;
  country: string;
  type?: string;
  isExternal?: boolean;
  isFavorite?: boolean;
  createdAt?: string;
  notes?: string;
  position?: [number, number];
  timestamp?: string;
}

export interface MapProps {
  locations: Location[];
  selectedLocation: Location | null;
  onLocationSelect: (location: Location) => void;
}

export interface SearchHistory {
  query: string;
  timestamp: string;
  results: number;
}
