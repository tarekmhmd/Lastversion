"""
SmartAiCity - Train/Validation Split Script
===========================================
Split preprocessed datasets into training (80%) and validation (20%) sets.

Usage:
    python scripts/split_train_val.py
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/split_data.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = BASE_DIR / 'datasets' / 'processed'
TRAINING_DIR = BASE_DIR / 'datasets' / 'training'
VALIDATION_DIR = BASE_DIR / 'datasets' / 'validation'

# Create directories
TRAINING_DIR.mkdir(parents=True, exist_ok=True)
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
(BASE_DIR / 'logs').mkdir(exist_ok=True)


def split_crime_data():
    """Split crime CSV data"""
    logger.info("=" * 50)
    logger.info("SPLITTING CRIME DATA")
    logger.info("=" * 50)

    crime_dir = PROCESSED_DATA_DIR / 'crime'
    csv_file = crime_dir / 'crime_processed.csv'

    if not csv_file.exists():
        logger.warning(f"Crime data not found: {csv_file}")
        return

    try:
        # Load data
        df = pd.read_csv(csv_file)
        logger.info(f"Loaded {len(df)} crime records")

        # Split data
        train_df, val_df = train_test_split(
            df,
            test_size=0.2,
            random_state=42,
            stratify=df['crime_type'] if 'crime_type' in df.columns else None
        )

        # Create output directories
        train_crime_dir = TRAINING_DIR / 'crime'
        val_crime_dir = VALIDATION_DIR / 'crime'
        train_crime_dir.mkdir(parents=True, exist_ok=True)
        val_crime_dir.mkdir(parents=True, exist_ok=True)

        # Save splits
        train_df.to_csv(train_crime_dir / 'crime_train.csv', index=False)
        val_df.to_csv(val_crime_dir / 'crime_val.csv', index=False)

        logger.info(f"✓ Training set: {len(train_df)} records")
        logger.info(f"✓ Validation set: {len(val_df)} records")

        # Copy metadata and encoders
        if (crime_dir / 'metadata.json').exists():
            shutil.copy(crime_dir / 'metadata.json', train_crime_dir / 'metadata.json')
        if (crime_dir / 'crime_type_encoder.pkl').exists():
            shutil.copy(crime_dir / 'crime_type_encoder.pkl', train_crime_dir / 'crime_type_encoder.pkl')

        # Save split info
        split_info = {
            'train_size': len(train_df),
            'val_size': len(val_df),
            'split_ratio': 0.8,
            'stratified': 'crime_type' in df.columns
        }
        with open(train_crime_dir / 'split_info.json', 'w') as f:
            json.dump(split_info, f, indent=2)

    except Exception as e:
        logger.error(f"✗ Error splitting crime data: {str(e)}")


def split_audio_data():
    """Split audio numpy arrays"""
    logger.info("=" * 50)
    logger.info("SPLITTING AUDIO DATA")
    logger.info("=" * 50)

    audio_dir = PROCESSED_DATA_DIR / 'audio'
    features_file = audio_dir / 'audio_features.npy'
    labels_file = audio_dir / 'audio_labels.npy'

    if not features_file.exists() or not labels_file.exists():
        logger.warning(f"Audio data not found in {audio_dir}")
        return

    try:
        # Load data
        features = np.load(features_file)
        labels = np.load(labels_file, allow_pickle=True)
        logger.info(f"Loaded {len(features)} audio samples")

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            features,
            labels,
            test_size=0.2,
            random_state=42,
            stratify=labels
        )

        # Create output directories
        train_audio_dir = TRAINING_DIR / 'audio'
        val_audio_dir = VALIDATION_DIR / 'audio'
        train_audio_dir.mkdir(parents=True, exist_ok=True)
        val_audio_dir.mkdir(parents=True, exist_ok=True)

        # Save splits
        np.save(train_audio_dir / 'audio_features_train.npy', X_train)
        np.save(train_audio_dir / 'audio_labels_train.npy', y_train)
        np.save(val_audio_dir / 'audio_features_val.npy', X_val)
        np.save(val_audio_dir / 'audio_labels_val.npy', y_val)

        logger.info(f"✓ Training set: {len(X_train)} samples")
        logger.info(f"✓ Validation set: {len(X_val)} samples")

        # Copy metadata
        if (audio_dir / 'metadata.json').exists():
            shutil.copy(audio_dir / 'metadata.json', train_audio_dir / 'metadata.json')

        # Save split info
        split_info = {
            'train_size': len(X_train),
            'val_size': len(X_val),
            'feature_dim': X_train.shape[1],
            'split_ratio': 0.8,
            'unique_labels': np.unique(labels).tolist()
        }
        with open(train_audio_dir / 'split_info.json', 'w') as f:
            json.dump(split_info, f, indent=2)

    except Exception as e:
        logger.error(f"✗ Error splitting audio data: {str(e)}")


def split_video_data():
    """Split video numpy arrays"""
    logger.info("=" * 50)
    logger.info("SPLITTING VIDEO DATA")
    logger.info("=" * 50)

    video_dir = PROCESSED_DATA_DIR / 'video'
    frames_file = video_dir / 'video_frames.npy'
    labels_file = video_dir / 'video_labels.npy'

    if not frames_file.exists() or not labels_file.exists():
        logger.warning(f"Video data not found in {video_dir}")
        return

    try:
        # Load data
        frames = np.load(frames_file)
        labels = np.load(labels_file, allow_pickle=True)
        logger.info(f"Loaded {len(frames)} video samples")

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            frames,
            labels,
            test_size=0.2,
            random_state=42,
            stratify=labels
        )

        # Create output directories
        train_video_dir = TRAINING_DIR / 'video'
        val_video_dir = VALIDATION_DIR / 'video'
        train_video_dir.mkdir(parents=True, exist_ok=True)
        val_video_dir.mkdir(parents=True, exist_ok=True)

        # Save splits
        np.save(train_video_dir / 'video_frames_train.npy', X_train)
        np.save(train_video_dir / 'video_labels_train.npy', y_train)
        np.save(val_video_dir / 'video_frames_val.npy', X_val)
        np.save(val_video_dir / 'video_labels_val.npy', y_val)

        logger.info(f"✓ Training set: {len(X_train)} samples")
        logger.info(f"✓ Validation set: {len(X_val)} samples")

        # Copy metadata
        if (video_dir / 'metadata.json').exists():
            shutil.copy(video_dir / 'metadata.json', train_video_dir / 'metadata.json')

        # Save split info
        split_info = {
            'train_size': len(X_train),
            'val_size': len(X_val),
            'frames_per_video': X_train.shape[1],
            'frame_shape': list(X_train.shape[2:]),
            'split_ratio': 0.8,
            'unique_labels': np.unique(labels).tolist()
        }
        with open(train_video_dir / 'split_info.json', 'w') as f:
            json.dump(split_info, f, indent=2)

    except Exception as e:
        logger.error(f"✗ Error splitting video data: {str(e)}")


def split_nlp_data():
    """Split NLP text data"""
    logger.info("=" * 50)
    logger.info("SPLITTING NLP DATA")
    logger.info("=" * 50)

    nlp_dir = PROCESSED_DATA_DIR / 'nlp'
    csv_file = nlp_dir / 'nlp_processed.csv'

    if not csv_file.exists():
        logger.warning(f"NLP data not found: {csv_file}")
        return

    try:
        # Load data
        df = pd.read_csv(csv_file)
        logger.info(f"Loaded {len(df)} text samples")

        # Split data
        train_df, val_df = train_test_split(
            df,
            test_size=0.2,
            random_state=42,
            stratify=df['label'] if 'label' in df.columns else None
        )

        # Create output directories
        train_nlp_dir = TRAINING_DIR / 'nlp'
        val_nlp_dir = VALIDATION_DIR / 'nlp'
        train_nlp_dir.mkdir(parents=True, exist_ok=True)
        val_nlp_dir.mkdir(parents=True, exist_ok=True)

        # Save splits
        train_df.to_csv(train_nlp_dir / 'nlp_train.csv', index=False)
        val_df.to_csv(val_nlp_dir / 'nlp_val.csv', index=False)

        logger.info(f"✓ Training set: {len(train_df)} samples")
        logger.info(f"✓ Validation set: {len(val_df)} samples")

        # Copy metadata and vocabulary
        if (nlp_dir / 'metadata.json').exists():
            shutil.copy(nlp_dir / 'metadata.json', train_nlp_dir / 'metadata.json')
        if (nlp_dir / 'vocabulary.json').exists():
            shutil.copy(nlp_dir / 'vocabulary.json', train_nlp_dir / 'vocabulary.json')

        # Save split info
        split_info = {
            'train_size': len(train_df),
            'val_size': len(val_df),
            'split_ratio': 0.8,
            'unique_labels': df['label'].unique().tolist() if 'label' in df.columns else []
        }
        with open(train_nlp_dir / 'split_info.json', 'w') as f:
            json.dump(split_info, f, indent=2)

    except Exception as e:
        logger.error(f"✗ Error splitting NLP data: {str(e)}")


def split_vision_data():
    """Split vision/image data"""
    logger.info("=" * 50)
    logger.info("SPLITTING VISION DATA")
    logger.info("=" * 50)

    vision_dir = PROCESSED_DATA_DIR / 'vision'
    images_file = vision_dir / 'images.npy'
    labels_file = vision_dir / 'labels.npy'

    if not images_file.exists() or not labels_file.exists():
        logger.warning(f"Vision data not found in {vision_dir}")
        return

    try:
        # Load data
        images = np.load(images_file)
        labels = np.load(labels_file, allow_pickle=True)
        logger.info(f"Loaded {len(images)} images")

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            images,
            labels,
            test_size=0.2,
            random_state=42,
            stratify=labels
        )

        # Create output directories
        train_vision_dir = TRAINING_DIR / 'vision'
        val_vision_dir = VALIDATION_DIR / 'vision'
        train_vision_dir.mkdir(parents=True, exist_ok=True)
        val_vision_dir.mkdir(parents=True, exist_ok=True)

        # Save splits
        np.save(train_vision_dir / 'images_train.npy', X_train)
        np.save(train_vision_dir / 'labels_train.npy', y_train)
        np.save(val_vision_dir / 'images_val.npy', X_val)
        np.save(val_vision_dir / 'labels_val.npy', y_val)

        logger.info(f"✓ Training set: {len(X_train)} images")
        logger.info(f"✓ Validation set: {len(X_val)} images")

        # Copy metadata
        if (vision_dir / 'metadata.json').exists():
            shutil.copy(vision_dir / 'metadata.json', train_vision_dir / 'metadata.json')

        # Save split info
        split_info = {
            'train_size': len(X_train),
            'val_size': len(X_val),
            'image_shape': list(X_train.shape[1:]),
            'split_ratio': 0.8,
            'unique_labels': np.unique(labels).tolist()
        }
        with open(train_vision_dir / 'split_info.json', 'w') as f:
            json.dump(split_info, f, indent=2)

    except Exception as e:
        logger.error(f"✗ Error splitting vision data: {str(e)}")


def main():
    """Main splitting function"""
    logger.info("Starting SmartAiCity Train/Validation Split")
    logger.info("=" * 50)

    try:
        # Split all datasets
        split_crime_data()
        split_audio_data()
        split_video_data()
        split_nlp_data()
        split_vision_data()

        logger.info("=" * 50)
        logger.info("✓ DATA SPLITTING COMPLETED SUCCESSFULLY")
        logger.info(f"✓ Training data saved to: {TRAINING_DIR}")
        logger.info(f"✓ Validation data saved to: {VALIDATION_DIR}")
        logger.info("=" * 50)

        # Print summary
        logger.info("\nSummary:")
        for data_type in ['crime', 'audio', 'video', 'nlp', 'vision']:
            train_dir = TRAINING_DIR / data_type
            if train_dir.exists() and (train_dir / 'split_info.json').exists():
                with open(train_dir / 'split_info.json', 'r') as f:
                    info = json.load(f)
                logger.info(f"  {data_type.upper()}: {info['train_size']} train, {info['val_size']} val")

    except Exception as e:
        logger.error(f"✗ Data splitting failed: {str(e)}")
        raise


if __name__ == '__main__':
    main()
