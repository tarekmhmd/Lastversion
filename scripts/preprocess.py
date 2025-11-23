"""
SmartAiCity - Data Preprocessing Script
========================================
This script preprocesses all datasets for the Smart City AI project:
- Crime data (CSV files)
- Audio files (WAV, MP3)
- Video files (MP4, AVI)
- NLP text data (CSV, TXT)
- Vision images (JPG, PNG)

Usage:
    python scripts/preprocess.py --all
    python scripts/preprocess.py --crime
    python scripts/preprocess.py --audio --video
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Audio processing
import librosa
import soundfile as sf
from scipy import signal

# Video processing
import cv2

# NLP processing
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# Vision processing
from PIL import Image
from sklearn.preprocessing import StandardScaler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/preprocessing.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / 'datasets' / 'raw'
PROCESSED_DATA_DIR = BASE_DIR / 'datasets' / 'processed'

# Create directories
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
(BASE_DIR / 'logs').mkdir(exist_ok=True)


class CrimeDataPreprocessor:
    """Preprocess crime CSV data"""

    def __init__(self):
        self.output_dir = PROCESSED_DATA_DIR / 'crime'
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process(self):
        """Process all crime CSV files"""
        logger.info("=" * 50)
        logger.info("CRIME DATA PREPROCESSING")
        logger.info("=" * 50)

        crime_files = list((RAW_DATA_DIR / 'crime').glob('*.csv'))
        if not crime_files:
            logger.warning("No crime CSV files found in datasets/raw/crime/")
            return

        all_crime_data = []

        for file_path in crime_files:
            logger.info(f"Processing: {file_path.name}")
            try:
                df = pd.read_csv(file_path)
                df = self.clean_data(df)
                df = self.engineer_features(df)
                all_crime_data.append(df)
                logger.info(f"✓ Successfully processed {len(df)} records from {file_path.name}")
            except Exception as e:
                logger.error(f"✗ Error processing {file_path.name}: {str(e)}")

        if all_crime_data:
            # Combine all crime data
            combined_df = pd.concat(all_crime_data, ignore_index=True)
            output_path = self.output_dir / 'crime_processed.csv'
            combined_df.to_csv(output_path, index=False)
            logger.info(f"✓ Saved combined crime data: {output_path}")
            logger.info(f"  Total records: {len(combined_df)}")
            logger.info(f"  Features: {list(combined_df.columns)}")

            # Save feature metadata
            self._save_metadata(combined_df)

    def clean_data(self, df):
        """Clean crime data"""
        # Standardize column names
        df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')

        # Common column mappings
        column_mapping = {
            'date': ['date', 'incident_date', 'crime_date', 'timestamp'],
            'latitude': ['latitude', 'lat', 'y_coord'],
            'longitude': ['longitude', 'lon', 'lng', 'x_coord'],
            'crime_type': ['crime_type', 'type', 'offense', 'category'],
            'location': ['location', 'address', 'place', 'area']
        }

        # Rename columns
        for target_col, possible_names in column_mapping.items():
            for possible_name in possible_names:
                if possible_name in df.columns:
                    df.rename(columns={possible_name: target_col}, inplace=True)
                    break

        # Handle missing values
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date'])

        # Fill missing coordinates with mean
        if 'latitude' in df.columns:
            df['latitude'].fillna(df['latitude'].mean(), inplace=True)
        if 'longitude' in df.columns:
            df['longitude'].fillna(df['longitude'].mean(), inplace=True)

        # Fill missing categorical values
        if 'crime_type' in df.columns:
            df['crime_type'].fillna('unknown', inplace=True)
        if 'location' in df.columns:
            df['location'].fillna('unknown', inplace=True)

        # Remove duplicates
        df = df.drop_duplicates()

        return df

    def engineer_features(self, df):
        """Create features for crime prediction"""
        # Time-based features
        if 'date' in df.columns:
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['day'] = df['date'].dt.day
            df['day_of_week'] = df['date'].dt.dayofweek
            df['hour'] = df['date'].dt.hour
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

            # Time of day categories
            df['time_of_day'] = pd.cut(
                df['hour'],
                bins=[0, 6, 12, 18, 24],
                labels=['night', 'morning', 'afternoon', 'evening']
            )

        # Encode crime types
        if 'crime_type' in df.columns:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            df['crime_type_encoded'] = le.fit_transform(df['crime_type'])

            # Save label encoder
            import pickle
            with open(self.output_dir / 'crime_type_encoder.pkl', 'wb') as f:
                pickle.dump(le, f)

        # Location-based features
        if 'latitude' in df.columns and 'longitude' in df.columns:
            # Create location zones
            df['lat_zone'] = pd.cut(df['latitude'], bins=10, labels=False)
            df['lon_zone'] = pd.cut(df['longitude'], bins=10, labels=False)
            df['location_zone'] = df['lat_zone'].astype(str) + '_' + df['lon_zone'].astype(str)

        return df

    def _save_metadata(self, df):
        """Save metadata about processed data"""
        metadata = {
            'n_records': len(df),
            'columns': list(df.columns),
            'date_range': {
                'start': str(df['date'].min()) if 'date' in df.columns else None,
                'end': str(df['date'].max()) if 'date' in df.columns else None
            },
            'crime_types': df['crime_type'].unique().tolist() if 'crime_type' in df.columns else [],
            'processed_at': datetime.now().isoformat()
        }

        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)


class AudioDataPreprocessor:
    """Preprocess audio files"""

    def __init__(self):
        self.output_dir = PROCESSED_DATA_DIR / 'audio'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_rate = 22050
        self.duration = 5  # seconds

    def process(self):
        """Process all audio files"""
        logger.info("=" * 50)
        logger.info("AUDIO DATA PREPROCESSING")
        logger.info("=" * 50)

        audio_dir = RAW_DATA_DIR / 'audio'
        audio_files = []
        for ext in ['*.wav', '*.mp3', '*.flac', '*.ogg']:
            audio_files.extend(list(audio_dir.glob(ext)))

        if not audio_files:
            logger.warning("No audio files found in datasets/raw/audio/")
            return

        features_list = []
        labels_list = []

        for file_path in audio_files:
            logger.info(f"Processing: {file_path.name}")
            try:
                features = self.extract_features(file_path)
                features_list.append(features)

                # Extract label from filename (e.g., gunshot_001.wav -> gunshot)
                label = file_path.stem.split('_')[0]
                labels_list.append(label)

                logger.info(f"✓ Extracted features: shape {features.shape}")
            except Exception as e:
                logger.error(f"✗ Error processing {file_path.name}: {str(e)}")

        if features_list:
            # Save as numpy arrays
            features_array = np.array(features_list)
            labels_array = np.array(labels_list)

            np.save(self.output_dir / 'audio_features.npy', features_array)
            np.save(self.output_dir / 'audio_labels.npy', labels_array)

            logger.info(f"✓ Saved audio features: {features_array.shape}")
            logger.info(f"✓ Unique labels: {np.unique(labels_array)}")

            # Save metadata
            self._save_metadata(features_array, labels_array)

    def extract_features(self, file_path):
        """Extract audio features using librosa"""
        # Load audio
        y, sr = librosa.load(file_path, sr=self.sample_rate, duration=self.duration)

        # Pad or trim to fixed length
        target_length = self.sample_rate * self.duration
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)))
        else:
            y = y[:target_length]

        # Extract features
        features = []

        # 1. MFCC (Mel-frequency cepstral coefficients)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        features.extend(mfcc_mean)
        features.extend(mfcc_std)

        # 2. Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)

        features.append(np.mean(spectral_centroid))
        features.append(np.std(spectral_centroid))
        features.append(np.mean(spectral_rolloff))
        features.append(np.std(spectral_rolloff))
        features.append(np.mean(spectral_bandwidth))
        features.append(np.std(spectral_bandwidth))

        # 3. Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)
        features.append(np.mean(zcr))
        features.append(np.std(zcr))

        # 4. RMS Energy
        rms = librosa.feature.rms(y=y)
        features.append(np.mean(rms))
        features.append(np.std(rms))

        # 5. Chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features.extend(np.mean(chroma, axis=1))

        return np.array(features)

    def _save_metadata(self, features, labels):
        """Save metadata"""
        metadata = {
            'n_samples': len(features),
            'feature_dim': features.shape[1],
            'sample_rate': self.sample_rate,
            'duration': self.duration,
            'labels': np.unique(labels).tolist(),
            'processed_at': datetime.now().isoformat()
        }

        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)


class VideoDataPreprocessor:
    """Preprocess video files"""

    def __init__(self):
        self.output_dir = PROCESSED_DATA_DIR / 'video'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frame_size = (224, 224)
        self.frames_per_video = 16

    def process(self):
        """Process all video files"""
        logger.info("=" * 50)
        logger.info("VIDEO DATA PREPROCESSING")
        logger.info("=" * 50)

        video_dir = RAW_DATA_DIR / 'video'
        video_files = []
        for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
            video_files.extend(list(video_dir.glob(ext)))

        if not video_files:
            logger.warning("No video files found in datasets/raw/video/")
            return

        features_list = []
        labels_list = []

        for file_path in video_files:
            logger.info(f"Processing: {file_path.name}")
            try:
                frames = self.extract_frames(file_path)
                features_list.append(frames)

                # Extract label from filename
                label = file_path.stem.split('_')[0]
                labels_list.append(label)

                logger.info(f"✓ Extracted {len(frames)} frames")
            except Exception as e:
                logger.error(f"✗ Error processing {file_path.name}: {str(e)}")

        if features_list:
            # Save as numpy arrays
            features_array = np.array(features_list)
            labels_array = np.array(labels_list)

            np.save(self.output_dir / 'video_frames.npy', features_array)
            np.save(self.output_dir / 'video_labels.npy', labels_array)

            logger.info(f"✓ Saved video frames: {features_array.shape}")
            logger.info(f"✓ Unique labels: {np.unique(labels_array)}")

            self._save_metadata(features_array, labels_array)

    def extract_frames(self, file_path):
        """Extract frames from video"""
        cap = cv2.VideoCapture(str(file_path))

        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate frame indices to sample
        if frame_count < self.frames_per_video:
            frame_indices = list(range(frame_count))
        else:
            frame_indices = np.linspace(0, frame_count - 1, self.frames_per_video, dtype=int)

        frames = []
        current_frame = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if current_frame in frame_indices:
                # Resize and normalize
                frame = cv2.resize(frame, self.frame_size)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype(np.float32) / 255.0
                frames.append(frame)

            current_frame += 1

            if len(frames) >= self.frames_per_video:
                break

        cap.release()

        # Pad if necessary
        while len(frames) < self.frames_per_video:
            frames.append(np.zeros((self.frame_size[0], self.frame_size[1], 3)))

        return np.array(frames[:self.frames_per_video])

    def _save_metadata(self, features, labels):
        """Save metadata"""
        metadata = {
            'n_videos': len(features),
            'frames_per_video': self.frames_per_video,
            'frame_size': self.frame_size,
            'labels': np.unique(labels).tolist(),
            'processed_at': datetime.now().isoformat()
        }

        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)


class NLPDataPreprocessor:
    """Preprocess NLP text data"""

    def __init__(self):
        self.output_dir = PROCESSED_DATA_DIR / 'nlp'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Download NLTK data if not present
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)

        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)

    def process(self):
        """Process all NLP text data"""
        logger.info("=" * 50)
        logger.info("NLP DATA PREPROCESSING")
        logger.info("=" * 50)

        nlp_dir = RAW_DATA_DIR / 'nlp'
        text_files = list(nlp_dir.glob('*.csv')) + list(nlp_dir.glob('*.txt'))

        if not text_files:
            logger.warning("No text files found in datasets/raw/nlp/")
            return

        all_texts = []
        all_labels = []

        for file_path in text_files:
            logger.info(f"Processing: {file_path.name}")
            try:
                if file_path.suffix == '.csv':
                    df = pd.read_csv(file_path)
                    # Assume columns: text, label
                    text_col = 'text' if 'text' in df.columns else df.columns[0]
                    label_col = 'label' if 'label' in df.columns else df.columns[1] if len(df.columns) > 1 else None

                    texts = df[text_col].tolist()
                    labels = df[label_col].tolist() if label_col else ['unknown'] * len(texts)
                else:
                    # Plain text file
                    with open(file_path, 'r', encoding='utf-8') as f:
                        texts = f.readlines()
                    labels = ['unknown'] * len(texts)

                # Process each text
                for text, label in zip(texts, labels):
                    processed_text = self.process_text(text)
                    all_texts.append(processed_text)
                    all_labels.append(label)

                logger.info(f"✓ Processed {len(texts)} texts")
            except Exception as e:
                logger.error(f"✗ Error processing {file_path.name}: {str(e)}")

        if all_texts:
            # Save processed data
            df = pd.DataFrame({
                'text': all_texts,
                'label': all_labels
            })

            output_path = self.output_dir / 'nlp_processed.csv'
            df.to_csv(output_path, index=False)

            logger.info(f"✓ Saved NLP data: {output_path}")
            logger.info(f"  Total texts: {len(df)}")
            logger.info(f"  Unique labels: {df['label'].unique()}")

            # Create vocabulary
            self._create_vocabulary(all_texts)
            self._save_metadata(df)

    def process_text(self, text):
        """Process individual text"""
        if pd.isna(text):
            return ""

        # Convert to string and lowercase
        text = str(text).lower()

        # Tokenize
        tokens = word_tokenize(text)

        # Remove stopwords and punctuation
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token.isalnum() and token not in stop_words]

        return ' '.join(tokens)

    def _create_vocabulary(self, texts):
        """Create vocabulary from texts"""
        from collections import Counter

        all_words = []
        for text in texts:
            all_words.extend(text.split())

        word_counts = Counter(all_words)
        vocab = {word: idx for idx, (word, _) in enumerate(word_counts.most_common(10000), 1)}
        vocab['<PAD>'] = 0
        vocab['<UNK>'] = len(vocab)

        # Save vocabulary
        with open(self.output_dir / 'vocabulary.json', 'w') as f:
            json.dump(vocab, f, indent=2)

        logger.info(f"✓ Created vocabulary: {len(vocab)} words")

    def _save_metadata(self, df):
        """Save metadata"""
        metadata = {
            'n_texts': len(df),
            'labels': df['label'].unique().tolist(),
            'avg_length': df['text'].str.split().str.len().mean(),
            'processed_at': datetime.now().isoformat()
        }

        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)


class VisionDataPreprocessor:
    """Preprocess vision/image data"""

    def __init__(self):
        self.output_dir = PROCESSED_DATA_DIR / 'vision'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.image_size = (224, 224)

    def process(self):
        """Process all image files"""
        logger.info("=" * 50)
        logger.info("VISION DATA PREPROCESSING")
        logger.info("=" * 50)

        vision_dir = RAW_DATA_DIR / 'vision'
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(list(vision_dir.glob(ext)))

        if not image_files:
            logger.warning("No image files found in datasets/raw/vision/")
            return

        images_list = []
        labels_list = []

        for file_path in image_files:
            logger.info(f"Processing: {file_path.name}")
            try:
                img_array = self.process_image(file_path)
                images_list.append(img_array)

                # Extract label from filename or parent directory
                label = file_path.parent.name if file_path.parent != vision_dir else file_path.stem.split('_')[0]
                labels_list.append(label)

            except Exception as e:
                logger.error(f"✗ Error processing {file_path.name}: {str(e)}")

        if images_list:
            # Save as numpy arrays
            images_array = np.array(images_list)
            labels_array = np.array(labels_list)

            np.save(self.output_dir / 'images.npy', images_array)
            np.save(self.output_dir / 'labels.npy', labels_array)

            logger.info(f"✓ Saved images: {images_array.shape}")
            logger.info(f"✓ Unique labels: {np.unique(labels_array)}")

            self._save_metadata(images_array, labels_array)

    def process_image(self, file_path):
        """Process individual image"""
        # Load image
        img = Image.open(file_path).convert('RGB')

        # Resize
        img = img.resize(self.image_size, Image.LANCZOS)

        # Convert to array and normalize
        img_array = np.array(img).astype(np.float32) / 255.0

        return img_array

    def _save_metadata(self, images, labels):
        """Save metadata"""
        metadata = {
            'n_images': len(images),
            'image_size': self.image_size,
            'labels': np.unique(labels).tolist(),
            'mean': images.mean(axis=(0, 1, 2)).tolist(),
            'std': images.std(axis=(0, 1, 2)).tolist(),
            'processed_at': datetime.now().isoformat()
        }

        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)


def main():
    """Main preprocessing function"""
    parser = argparse.ArgumentParser(description='Preprocess SmartAiCity datasets')
    parser.add_argument('--all', action='store_true', help='Process all datasets')
    parser.add_argument('--crime', action='store_true', help='Process crime data')
    parser.add_argument('--audio', action='store_true', help='Process audio data')
    parser.add_argument('--video', action='store_true', help='Process video data')
    parser.add_argument('--nlp', action='store_true', help='Process NLP data')
    parser.add_argument('--vision', action='store_true', help='Process vision data')

    args = parser.parse_args()

    # If no specific flag, process all
    if not any([args.crime, args.audio, args.video, args.nlp, args.vision]):
        args.all = True

    logger.info("Starting SmartAiCity Data Preprocessing")
    logger.info("=" * 50)

    # Create raw data directories if they don't exist
    for data_type in ['crime', 'audio', 'video', 'nlp', 'vision']:
        (RAW_DATA_DIR / data_type).mkdir(parents=True, exist_ok=True)

    try:
        if args.all or args.crime:
            preprocessor = CrimeDataPreprocessor()
            preprocessor.process()

        if args.all or args.audio:
            preprocessor = AudioDataPreprocessor()
            preprocessor.process()

        if args.all or args.video:
            preprocessor = VideoDataPreprocessor()
            preprocessor.process()

        if args.all or args.nlp:
            preprocessor = NLPDataPreprocessor()
            preprocessor.process()

        if args.all or args.vision:
            preprocessor = VisionDataPreprocessor()
            preprocessor.process()

        logger.info("=" * 50)
        logger.info("✓ PREPROCESSING COMPLETED SUCCESSFULLY")
        logger.info(f"✓ Processed data saved to: {PROCESSED_DATA_DIR}")
        logger.info("=" * 50)

    except Exception as e:
        logger.error(f"✗ Preprocessing failed: {str(e)}")
        raise


if __name__ == '__main__':
    main()
