'use client'

import dynamic from 'next/dynamic'
import React, { useState, useMemo, useCallback, useEffect } from 'react'
import { 
  MapPin, Search, Filter, Loader2, Plus, Star, Trash2, Upload, Download, Clock
} from 'lucide-react'
import { MapProps, Location, SearchHistory } from '../../types' // ملف واحد لجميع الأنواع

// Dynamic import للـ Leaflet map component
const MapComponent = dynamic<MapProps>(
  () => import('../../components/MapComponent').then(mod => mod.default),
  {
    ssr: false,
    loading: () => (
      <div className="w-full h-[500px] flex items-center justify-center bg-gray-100 rounded-lg">
        <div className="text-center">
          <Loader2 className="h-8 w-8 animate-spin text-indigo-600 mx-auto mb-2" />
          <p className="text-gray-600">جاري تحميل الخريطة...</p>
        </div>
      </div>
    ),
  }
)

export default function EnhancedMapsPage() {
  // -------------------------
  // States
  // -------------------------
  const [locations, setLocations] = useState<Location[]>(() => {
    if (typeof window !== 'undefined') {
      const saved = localStorage.getItem('map-locations')
      return saved ? JSON.parse(saved) : [
        { id: 1, name: 'القاهرة', lat: 30.0444, lng: 31.2357, type: 'capital', country: 'مصر', isFavorite: true },
        { id: 2, name: 'الرياض', lat: 24.7136, lng: 46.6753, type: 'capital', country: 'السعودية', isFavorite: true },
        { id: 3, name: 'دبي', lat: 25.2048, lng: 55.2708, type: 'commercial', country: 'الإمارات' },
        { id: 4, name: 'بيروت', lat: 33.8938, lng: 35.5018, type: 'capital', country: 'لبنان' },
        { id: 5, name: 'جدة', lat: 21.4858, lng: 39.1925, type: 'commercial', country: 'السعودية' },
        { id: 6, name: 'الدوحة', lat: 25.2854, lng: 51.5310, type: 'capital', country: 'قطر' },
      ]
    }
    return []
  })
  const [selectedLocation, setSelectedLocation] = useState<Location | null>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedType, setSelectedType] = useState<'all' | 'capital' | 'commercial' | 'external'>('all')
  const [showFavorites, setShowFavorites] = useState(false)
  const [showHistory, setShowHistory] = useState(false)
  const [externalResults, setExternalResults] = useState<Location[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [isSearchingExternal, setIsSearchingExternal] = useState(false)
  const [locationNotes, setLocationNotes] = useState<{ [key: string]: string }>({})
  const [searchHistory, setSearchHistory] = useState<SearchHistory[]>([])

  // -------------------------
  // Effects
  // -------------------------
  useEffect(() => {
    localStorage.setItem('map-locations', JSON.stringify(locations))
  }, [locations])

  useEffect(() => {
    if (typeof window !== 'undefined') {
      const saved = localStorage.getItem('map-search-history')
      if (saved) setSearchHistory(JSON.parse(saved))
    }
  }, [])

  // -------------------------
  // External search
  // -------------------------
  const searchExternalLocations = async (query: string) => {
    if (!query.trim() || query.length < 2) return setExternalResults([])
    setIsSearchingExternal(true)
    try {
      const res = await fetch(`https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(query)}&accept-language=ar&limit=10`)
      const data = await res.json()
      const results: Location[] = data.map((place: any, idx: number) => ({
        id: `external-${Date.now()}-${idx}`,
        name: place.display_name.split(',')[0],
        lat: parseFloat(place.lat),
        lng: parseFloat(place.lon),
        type: 'external',
        country: place.display_name.split(',').slice(-1)[0]?.trim() || 'غير معروف',
        isExternal: true
      }))
      setExternalResults(results)

      // سجل البحث
      const newSearch: SearchHistory = { query, timestamp: new Date().toISOString(), results: results.length }
      const updatedHistory = [newSearch, ...searchHistory.slice(0, 9)]
      setSearchHistory(updatedHistory)
      localStorage.setItem('map-search-history', JSON.stringify(updatedHistory))
    } catch (err) {
      console.error(err)
      setExternalResults([])
    } finally {
      setIsSearchingExternal(false)
    }
  }

  // -------------------------
  // Combined search
  // -------------------------
  const allLocations = useMemo(() => {
    let base = locations
    if (showFavorites) base = base.filter(l => l.isFavorite)
    const localResults = base.filter(l => {
      const query = searchQuery.trim().toLowerCase()
      const matchSearch = !query || l.name.toLowerCase().includes(query) || l.country.toLowerCase().includes(query)
      const matchType = selectedType === 'all' || (selectedType === 'external' ? l.isExternal : l.type === selectedType)
      return matchSearch && matchType
    })
    if (searchQuery.trim().length >= 2) {
      const filteredExternal = externalResults.filter(l => selectedType === 'all' || selectedType === 'external')
      return [...localResults, ...filteredExternal]
    }
    return localResults
  }, [locations, externalResults, searchQuery, selectedType, showFavorites])

  // -------------------------
  // Handlers
  // -------------------------
  const handleLocationSelect = useCallback((loc: Location) => {
    setSelectedLocation(loc)
    if (loc.isExternal && !locations.some(l => l.id === loc.id)) {
      setLocations(prev => [...prev, { ...loc, isExternal: false, createdAt: new Date().toISOString() }])
    }
  }, [locations])

  const toggleFavorite = useCallback((id: string | number, e?: React.MouseEvent) => {
    e?.stopPropagation()
    setLocations(prev => prev.map(l => l.id === id ? { ...l, isFavorite: !l.isFavorite } : l))
  }, [])

  const deleteLocation = useCallback((id: string | number, e: React.MouseEvent) => {
    e.stopPropagation()
    setLocations(prev => prev.filter(l => l.id !== id))
    if (selectedLocation?.id === id) setSelectedLocation(null)
  }, [selectedLocation])

  const updateLocationNotes = useCallback((id: string | number, notes: string) => {
    setLocationNotes(prev => ({ ...prev, [id.toString()]: notes }))
  }, [])

  const exportLocations = useCallback(() => {
    const data = { locations, exportDate: new Date().toISOString(), version: '1.0' }
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `map-locations-${new Date().toISOString().split('T')[0]}.json`
    a.click()
    URL.revokeObjectURL(url)
  }, [locations])

  const importLocations = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = (ev) => {
      try {
        const data = JSON.parse(ev.target?.result as string)
        if (data.locations && Array.isArray(data.locations)) {
          setLocations(prev => [...prev, ...data.locations])
          alert(`تم استيراد ${data.locations.length} موقع بنجاح`)
        }
      } catch {
        alert('خطأ في الملف. تأكد من صحته.')
      }
    }
    reader.readAsText(file)
    e.target.value = ''
  }, [])

  const clearSearchHistory = useCallback(() => {
    setSearchHistory([])
    localStorage.removeItem('map-search-history')
  }, [])

  // -------------------------
  // Render
  // -------------------------
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* العنوان والتحكم */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-indigo-100 rounded-full mb-4">
            <MapPin className="h-8 w-8 text-indigo-600" />
          </div>
          <h1 className="text-4xl font-bold text-gray-900 mb-3">نظام الخرائط الذكية</h1>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto leading-relaxed mb-6">
            استكشف وتحليل البيانات الجغرافية باستخدام أحدث التقنيات في الخرائط التفاعلية
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          {/* الشريط الجانبي */}
          <div className="bg-white rounded-2xl shadow-lg border border-gray-100 p-6 lg:sticky lg:top-8 lg:h-fit">
            {/* ... جميع عناصر البحث والتصفية وسجل البحث كما في كودك ... */}
          </div>

          {/* قسم الخريطة */}
          <div className="lg:col-span-3">
            <div className="h-96 sm:h-[500px]">
              <MapComponent
                locations={allLocations}
                selectedLocation={selectedLocation}
                onLocationSelect={handleLocationSelect}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
