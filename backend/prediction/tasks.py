from celery import shared_task
import pandas as pd
import numpy as np
from .models import CrimePrediction, HotZone
from vision.models import IncidentDetection
from audio.models import AudioDetection
import logging
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import os
from django.conf import settings
from django.db.models import Count
from collections import defaultdict

logger = logging.getLogger(__name__)

# Global model cache
_crime_model = None
_label_encoders = {}

def train_crime_prediction_model():
    """
    Train a simple crime prediction model using historical data
    """
    try:
        # Gather historical incident data
        incidents = IncidentDetection.objects.filter(is_verified=True).values(
            'incident_type', 'camera__location', 'timestamp', 'priority'
        )

        if incidents.count() < 10:
            logger.warning("Not enough data to train model, using default predictions")
            return None

        # Create DataFrame
        df = pd.DataFrame(list(incidents))

        # Feature engineering
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        df['month'] = pd.to_datetime(df['timestamp']).dt.month

        # Encode categorical features
        le_location = LabelEncoder()
        le_type = LabelEncoder()

        df['location_encoded'] = le_location.fit_transform(df['camera__location'])
        df['type_encoded'] = le_type.fit_transform(df['incident_type'])

        # Prepare features and target
        X = df[['location_encoded', 'hour', 'day_of_week', 'month']]
        y = df['type_encoded']

        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)

        # Cache label encoders
        _label_encoders['location'] = le_location
        _label_encoders['type'] = le_type

        logger.info("Crime prediction model trained successfully")
        return model

    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        return None

def get_crime_model():
    """
    Get or train crime prediction model
    """
    global _crime_model
    if _crime_model is None:
        _crime_model = train_crime_prediction_model()
    return _crime_model

@shared_task
def update_crime_predictions():
    """
    Update crime predictions based on new data
    """
    try:
        logger.info("Starting crime prediction update")

        # Train/update model
        model = get_crime_model()

        # Locations for Mansoura
        locations = [
            {'name': 'وسط المنصورة', 'lat': 31.0409, 'lon': 31.3785},
            {'name': 'جامعة المنصورة', 'lat': 31.0425, 'lon': 31.3550},
            {'name': 'المنطقة الصناعية', 'lat': 31.0380, 'lon': 31.3900},
            {'name': 'شارع الجلاء', 'lat': 31.0450, 'lon': 31.3800},
            {'name': 'حي توريل', 'lat': 31.0550, 'lon': 31.3950},
        ]

        crime_types = ['theft', 'assault', 'vandalism', 'traffic', 'public_order']

        # Get historical incident statistics
        incident_stats = defaultdict(lambda: defaultdict(int))
        historical_incidents = IncidentDetection.objects.filter(
            timestamp__gte=datetime.now() - timedelta(days=30)
        )

        for incident in historical_incidents:
            location = incident.camera.location
            inc_type = incident.incident_type
            incident_stats[location][inc_type] += 1

        new_predictions = []

        # Generate predictions for next 7 days
        for day_offset in range(7):
            predicted_date = datetime.now().date() + timedelta(days=day_offset)

            for location in locations:
                for crime_type in crime_types:
                    # Calculate probability based on historical data
                    location_name = location['name']
                    base_probability = 0.1

                    # Increase probability based on historical incidents
                    if location_name in incident_stats:
                        incident_count = incident_stats[location_name].get(crime_type, 0)
                        base_probability = min(0.3 + (incident_count * 0.05), 0.85)

                    # Time window prediction (high risk hours)
                    time_window = "18:00-22:00" if crime_type in ['theft', 'assault'] else "12:00-18:00"

                    # Confidence calculation
                    confidence = 0.65 + (base_probability * 0.2)

                    # Factors influencing prediction
                    factors = {
                        'historical_incidents': incident_stats[location_name].get(crime_type, 0),
                        'day_of_week': predicted_date.weekday(),
                        'weather': 'clear',
                        'events': []
                    }

                    prediction = CrimePrediction(
                        location=location_name,
                        latitude=location['lat'],
                        longitude=location['lon'],
                        crime_type=crime_type,
                        probability=base_probability,
                        predicted_date=predicted_date,
                        time_window=time_window,
                        confidence=confidence,
                        factors=factors
                    )
                    new_predictions.append(prediction)

        # Delete old predictions and save new ones
        CrimePrediction.objects.filter(
            predicted_date__lt=datetime.now().date()
        ).delete()

        CrimePrediction.objects.bulk_create(new_predictions, ignore_conflicts=True)

        logger.info(f"Created {len(new_predictions)} new predictions")
        return f"Crime predictions updated - {len(new_predictions)} predictions"

    except Exception as e:
        logger.error(f"Error updating crime predictions: {str(e)}", exc_info=True)
        return f"Error: {str(e)}"

@shared_task
def update_hot_zones():
    """
    Update hot zones based on recent incident data
    """
    try:
        logger.info("Starting hot zone update")

        # Get incidents from last 30 days
        recent_incidents = IncidentDetection.objects.filter(
            timestamp__gte=datetime.now() - timedelta(days=30)
        ).values('camera__location', 'camera__latitude', 'camera__longitude').annotate(
            incident_count=Count('id')
        )

        # Define hot zones for Mansoura
        hot_zones_data = []

        for incident_data in recent_incidents:
            location = incident_data['camera__location']
            count = incident_data['incident_count']
            lat = incident_data['camera__latitude']
            lon = incident_data['camera__longitude']

            # Determine risk level based on incident count
            if count >= 15:
                risk_level = 'high'
            elif count >= 8:
                risk_level = 'medium'
            else:
                risk_level = 'low'

            hot_zones_data.append({
                'name': location,
                'lat': lat,
                'lon': lon,
                'risk': risk_level,
                'incidents': count
            })

        # Add default zones if no data
        if not hot_zones_data:
            hot_zones_data = [
                {'name': 'وسط المنصورة', 'lat': 31.0409, 'lon': 31.3785, 'risk': 'medium', 'incidents': 10},
                {'name': 'جامعة المنصورة', 'lat': 31.0425, 'lon': 31.3550, 'risk': 'low', 'incidents': 5},
                {'name': 'حي توريل', 'lat': 31.0550, 'lon': 31.3950, 'risk': 'low', 'incidents': 3},
            ]

        # Update or create hot zones
        for zone_data in hot_zones_data:
            HotZone.objects.update_or_create(
                name=zone_data['name'],
                defaults={
                    'latitude': zone_data['lat'],
                    'longitude': zone_data['lon'],
                    'radius': 500.0,
                    'risk_level': zone_data['risk'],
                    'incident_count': zone_data['incidents'],
                    'description': f"Zone {zone_data['name']} - Risk: {zone_data['risk']}"
                }
            )

        logger.info(f"Updated {len(hot_zones_data)} hot zones")
        return f"Hot zones updated - {len(hot_zones_data)} zones"

    except Exception as e:
        logger.error(f"Error updating hot zones: {str(e)}", exc_info=True)
        return f"Error: {str(e)}"

@shared_task
def analyze_crime_patterns():
    """
    Analyze crime patterns using AI/ML
    """
    try:
        logger.info("Starting crime pattern analysis")

        # Get recent incidents
        incidents = IncidentDetection.objects.filter(
            timestamp__gte=datetime.now() - timedelta(days=90)
        )

        # Analyze by time of day
        time_patterns = defaultdict(int)
        for incident in incidents:
            hour = incident.timestamp.hour
            time_slot = f"{hour:02d}:00"
            time_patterns[time_slot] += 1

        # Analyze by type
        type_patterns = incidents.values('incident_type').annotate(
            count=Count('id')
        ).order_by('-count')

        # Analyze by location
        location_patterns = incidents.values('camera__location').annotate(
            count=Count('id')
        ).order_by('-count')

        patterns = {
            'time_patterns': dict(time_patterns),
            'type_patterns': list(type_patterns),
            'location_patterns': list(location_patterns),
            'total_incidents': incidents.count()
        }

        logger.info(f"Crime pattern analysis complete: {patterns['total_incidents']} incidents analyzed")
        return patterns

    except Exception as e:
        logger.error(f"Error analyzing crime patterns: {str(e)}", exc_info=True)
        return {"error": str(e)}
