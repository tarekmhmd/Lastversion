from celery import shared_task
import librosa
import numpy as np
from .models import AudioDetection
import logging
import os
from django.conf import settings

logger = logging.getLogger(__name__)

def extract_audio_features(audio_path):
    """
    Extract audio features using librosa
    """
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=22050)

        # Extract features
        features = {}

        # MFCC (Mel-frequency cepstral coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features['mfcc_mean'] = np.mean(mfccs, axis=1)
        features['mfcc_std'] = np.std(mfccs, axis=1)

        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        features['spectral_centroid_mean'] = np.mean(spectral_centroid)
        features['spectral_centroid_std'] = np.std(spectral_centroid)

        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)

        # RMS Energy
        rms = librosa.feature.rms(y=y)
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)

        # Spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features['rolloff_mean'] = np.mean(rolloff)

        # Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = tempo

        return features
    except Exception as e:
        logger.error(f"Error extracting audio features: {str(e)}")
        return None

def classify_sound(features):
    """
    Classify sound based on extracted features
    Uses rule-based classification (can be replaced with ML model)
    """
    if features is None:
        return 'unknown', 0.5

    # Rule-based classification based on audio characteristics
    rms_mean = features.get('rms_mean', 0)
    zcr_mean = features.get('zcr_mean', 0)
    spectral_centroid_mean = features.get('spectral_centroid_mean', 0)
    tempo = features.get('tempo', 0)

    # High energy + high ZCR = explosion or gunshot
    if rms_mean > 0.15 and zcr_mean > 0.1:
        if spectral_centroid_mean > 3000:
            return 'gunshot', 0.85
        else:
            return 'explosion', 0.80

    # High spectral centroid = scream or alarm
    elif spectral_centroid_mean > 2500:
        if tempo > 150:
            return 'alarm', 0.75
        else:
            return 'scream', 0.78

    # Medium energy with sharp transients = crash
    elif rms_mean > 0.08 and zcr_mean > 0.05:
        return 'crash', 0.70

    # High tempo + moderate energy = horn/siren
    elif tempo > 120:
        return 'horn', 0.72

    # Low confidence for normal sounds
    else:
        return 'normal', 0.60

@shared_task
def analyze_audio_file(audio_detection_id):
    """
    Analyze audio file to detect dangerous sounds
    """
    try:
        audio_detection = AudioDetection.objects.get(id=audio_detection_id)
        logger.info(f"Starting audio analysis: {audio_detection.audio_file.name}")

        # Get audio file path
        audio_path = audio_detection.audio_file.path

        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            return "Audio file not found"

        # Extract features
        features = extract_audio_features(audio_path)

        if features is None:
            logger.error("Failed to extract audio features")
            return "Failed to extract features"

        # Classify sound
        detected_sound, confidence = classify_sound(features)

        # Determine priority based on sound type
        critical_sounds = ['gunshot', 'explosion', 'scream']
        high_priority_sounds = ['crash', 'alarm', 'horn']

        if detected_sound in critical_sounds:
            priority = 'critical'
        elif detected_sound in high_priority_sounds:
            priority = 'high'
        else:
            priority = 'medium'

        # Update audio detection
        audio_detection.sound_type = detected_sound
        audio_detection.confidence = confidence
        audio_detection.priority = priority
        audio_detection.save()

        logger.info(f"Detected sound: {detected_sound} with confidence: {confidence:.2f}")
        return f"Audio analysis complete - Detected: {detected_sound}"

    except AudioDetection.DoesNotExist:
        logger.error(f"Audio detection not found: {audio_detection_id}")
        return "Audio detection not found"
    except Exception as e:
        logger.error(f"Error analyzing audio file: {str(e)}", exc_info=True)
        return f"Error: {str(e)}"

@shared_task
def process_real_time_audio():
    """
    Process real-time audio from sensors
    """
    logger.info("Starting real-time audio processing")
    # Implement real-time audio stream processing
    return "Real-time audio processing active"

@shared_task
def batch_process_audio_files():
    """
    Process unprocessed audio files in batch
    """
    unprocessed_audio = AudioDetection.objects.filter(sound_type__isnull=True)
    count = unprocessed_audio.count()

    for audio_detection in unprocessed_audio:
        analyze_audio_file.delay(audio_detection.id)

    logger.info(f"Queued {count} audio files for processing")
    return f"Queued {count} audio files for processing"
