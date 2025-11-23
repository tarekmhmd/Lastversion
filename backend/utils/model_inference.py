"""
SmartAiCity - Model Inference Utilities
=======================================
Load trained models and provide inference functions for Django views.
"""

import os
import pickle
import numpy as np
import torch
import json
from pathlib import Path
from django.conf import settings

# Base paths
BASE_DIR = Path(settings.BASE_DIR).parent
MODEL_DIR = BASE_DIR / 'models' / 'checkpoints'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Global model cache
_MODELS = {}


def load_crime_model():
    """Load crime prediction model"""
    if 'crime' not in _MODELS:
        try:
            with open(MODEL_DIR / 'crime_model.pkl', 'rb') as f:
                model = pickle.load(f)
            with open(MODEL_DIR / 'crime_scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
            with open(MODEL_DIR / 'crime_label_encoder.pkl', 'rb') as f:
                label_encoder = pickle.load(f)

            _MODELS['crime'] = {
                'model': model,
                'scaler': scaler,
                'label_encoder': label_encoder
            }
        except Exception as e:
            raise Exception(f"Error loading crime model: {str(e)}")

    return _MODELS['crime']


def load_audio_model():
    """Load audio classification model"""
    if 'audio' not in _MODELS:
        try:
            from train import AudioCNN

            with open(MODEL_DIR / 'audio_label_encoder.pkl', 'rb') as f:
                label_encoder = pickle.load(f)

            # Load metadata to get input dimensions
            with open(BASE_DIR / 'datasets' / 'processed' / 'audio' / 'metadata.json', 'r') as f:
                metadata = json.load(f)

            input_dim = metadata['feature_dim']
            num_classes = len(label_encoder.classes_)

            model = AudioCNN(input_dim, num_classes).to(device)
            model.load_state_dict(torch.load(MODEL_DIR / 'audio_model.pth', map_location=device))
            model.eval()

            _MODELS['audio'] = {
                'model': model,
                'label_encoder': label_encoder
            }
        except Exception as e:
            raise Exception(f"Error loading audio model: {str(e)}")

    return _MODELS['audio']


def load_video_model():
    """Load video classification model"""
    if 'video' not in _MODELS:
        try:
            from train import Video3DCNN

            with open(MODEL_DIR / 'video_label_encoder.pkl', 'rb') as f:
                label_encoder = pickle.load(f)

            num_classes = len(label_encoder.classes_)
            model = Video3DCNN(num_classes).to(device)
            model.load_state_dict(torch.load(MODEL_DIR / 'video_model.pth', map_location=device))
            model.eval()

            _MODELS['video'] = {
                'model': model,
                'label_encoder': label_encoder
            }
        except Exception as e:
            raise Exception(f"Error loading video model: {str(e)}")

    return _MODELS['video']


def load_nlp_model():
    """Load NLP model"""
    if 'nlp' not in _MODELS:
        try:
            from train import TextLSTM

            with open(MODEL_DIR / 'nlp_label_encoder.pkl', 'rb') as f:
                label_encoder = pickle.load(f)

            # Load vocabulary
            with open(BASE_DIR / 'datasets' / 'training' / 'nlp' / 'vocabulary.json', 'r') as f:
                vocab = json.load(f)

            vocab_size = len(vocab)
            num_classes = len(label_encoder.classes_)

            model = TextLSTM(vocab_size, 128, 256, num_classes).to(device)
            model.load_state_dict(torch.load(MODEL_DIR / 'nlp_model.pth', map_location=device))
            model.eval()

            _MODELS['nlp'] = {
                'model': model,
                'label_encoder': label_encoder,
                'vocab': vocab
            }
        except Exception as e:
            raise Exception(f"Error loading NLP model: {str(e)}")

    return _MODELS['nlp']


def load_vision_model():
    """Load vision classification model"""
    if 'vision' not in _MODELS:
        try:
            from train import VisionCNN

            with open(MODEL_DIR / 'vision_label_encoder.pkl', 'rb') as f:
                label_encoder = pickle.load(f)

            num_classes = len(label_encoder.classes_)
            model = VisionCNN(num_classes).to(device)
            model.load_state_dict(torch.load(MODEL_DIR / 'vision_model.pth', map_location=device))
            model.eval()

            _MODELS['vision'] = {
                'model': model,
                'label_encoder': label_encoder
            }
        except Exception as e:
            raise Exception(f"Error loading vision model: {str(e)}")

    return _MODELS['vision']


# ============================================================================
# INFERENCE FUNCTIONS
# ============================================================================

def predict_crime(location, latitude, longitude, date, hour):
    """
    Predict crime probability for a given location and time

    Args:
        location: str - Location name
        latitude: float - Latitude coordinate
        longitude: float - Longitude coordinate
        date: datetime - Date of prediction
        hour: int - Hour of day (0-23)

    Returns:
        dict with prediction results
    """
    try:
        crime_models = load_crime_model()
        model = crime_models['model']
        scaler = crime_models['scaler']
        label_encoder = crime_models['label_encoder']

        # Prepare features
        month = date.month
        day_of_week = date.weekday()
        is_weekend = 1 if day_of_week >= 5 else 0

        features = np.array([[latitude, longitude, month, day_of_week, hour, is_weekend]])
        features_scaled = scaler.transform(features)

        # Predict
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]

        # Get top 3 predictions
        top_indices = np.argsort(probabilities)[-3:][::-1]
        top_predictions = []

        for idx in top_indices:
            top_predictions.append({
                'crime_type': label_encoder.classes_[idx],
                'probability': float(probabilities[idx])
            })

        return {
            'success': True,
            'predicted_crime_type': label_encoder.classes_[prediction],
            'probability': float(probabilities[prediction]),
            'top_predictions': top_predictions,
            'location': location,
            'coordinates': {'lat': latitude, 'lon': longitude}
        }

    except Exception as e:
        return {'success': False, 'error': str(e)}


def predict_audio(audio_features):
    """
    Classify audio based on extracted features

    Args:
        audio_features: numpy array of audio features

    Returns:
        dict with classification results
    """
    try:
        audio_models = load_audio_model()
        model = audio_models['model']
        label_encoder = audio_models['label_encoder']

        # Predict
        with torch.no_grad():
            features_tensor = torch.FloatTensor(audio_features).unsqueeze(0).to(device)
            outputs = model(features_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()
            prediction = torch.argmax(outputs, dim=1).item()

        return {
            'success': True,
            'sound_type': label_encoder.classes_[prediction],
            'confidence': float(probabilities[prediction]),
            'all_probabilities': {
                label_encoder.classes_[i]: float(probabilities[i])
                for i in range(len(probabilities))
            }
        }

    except Exception as e:
        return {'success': False, 'error': str(e)}


def predict_video(video_frames):
    """
    Classify video based on extracted frames

    Args:
        video_frames: numpy array of video frames (frames, height, width, channels)

    Returns:
        dict with classification results
    """
    try:
        video_models = load_video_model()
        model = video_models['model']
        label_encoder = video_models['label_encoder']

        # Predict
        with torch.no_grad():
            frames_tensor = torch.FloatTensor(video_frames).unsqueeze(0).to(device)
            outputs = model(frames_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()
            prediction = torch.argmax(outputs, dim=1).item()

        return {
            'success': True,
            'event_type': label_encoder.classes_[prediction],
            'confidence': float(probabilities[prediction]),
            'all_probabilities': {
                label_encoder.classes_[i]: float(probabilities[i])
                for i in range(len(probabilities))
            }
        }

    except Exception as e:
        return {'success': False, 'error': str(e)}


def predict_text(text):
    """
    Classify text using NLP model

    Args:
        text: str - Input text

    Returns:
        dict with classification results
    """
    try:
        nlp_models = load_nlp_model()
        model = nlp_models['model']
        label_encoder = nlp_models['label_encoder']
        vocab = nlp_models['vocab']

        # Preprocess text
        tokens = text.lower().split()
        max_length = 100
        sequence = [vocab.get(token, vocab.get('<UNK>', 0)) for token in tokens]

        if len(sequence) < max_length:
            sequence += [0] * (max_length - len(sequence))
        else:
            sequence = sequence[:max_length]

        # Predict
        with torch.no_grad():
            text_tensor = torch.LongTensor([sequence]).to(device)
            outputs = model(text_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()
            prediction = torch.argmax(outputs, dim=1).item()

        return {
            'success': True,
            'category': label_encoder.classes_[prediction],
            'confidence': float(probabilities[prediction]),
            'all_probabilities': {
                label_encoder.classes_[i]: float(probabilities[i])
                for i in range(len(probabilities))
            }
        }

    except Exception as e:
        return {'success': False, 'error': str(e)}


def predict_image(image_array):
    """
    Classify image using vision model

    Args:
        image_array: numpy array of image (height, width, channels)

    Returns:
        dict with classification results
    """
    try:
        vision_models = load_vision_model()
        model = vision_models['model']
        label_encoder = vision_models['label_encoder']

        # Predict
        with torch.no_grad():
            image_tensor = torch.FloatTensor(image_array).unsqueeze(0).to(device)
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()
            prediction = torch.argmax(outputs, dim=1).item()

        return {
            'success': True,
            'class': label_encoder.classes_[prediction],
            'confidence': float(probabilities[prediction]),
            'all_probabilities': {
                label_encoder.classes_[i]: float(probabilities[i])
                for i in range(len(probabilities))
            }
        }

    except Exception as e:
        return {'success': False, 'error': str(e)}
