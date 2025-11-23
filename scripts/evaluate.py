"""
SmartAiCity - Model Evaluation Script
=====================================
Evaluate trained models and generate performance reports.

Usage:
    python scripts/evaluate.py --all
    python scripts/evaluate.py --crime --audio
"""

import os
import sys
import argparse
import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, confusion_matrix, classification_report)
import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/evaluation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
VALIDATION_DIR = BASE_DIR / 'datasets' / 'validation'
MODEL_DIR = BASE_DIR / 'models' / 'checkpoints'
REPORT_DIR = BASE_DIR / 'datasets' / 'reports'

# Create directories
REPORT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate_crime_model():
    """Evaluate crime prediction model"""
    logger.info("=" * 50)
    logger.info("EVALUATING CRIME PREDICTION MODEL")
    logger.info("=" * 50)

    try:
        # Load model
        with open(MODEL_DIR / 'crime_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open(MODEL_DIR / 'crime_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open(MODEL_DIR / 'crime_label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)

        # Load validation data
        val_df = pd.read_csv(VALIDATION_DIR / 'crime' / 'crime_val.csv')

        # Prepare features
        feature_cols = ['latitude', 'longitude', 'month', 'day_of_week', 'hour', 'is_weekend']
        feature_cols = [col for col in feature_cols if col in val_df.columns]

        X_val = val_df[feature_cols].values
        y_val = val_df['crime_type_encoded'].values

        # Scale
        X_val = scaler.transform(X_val)

        # Predict
        y_pred = model.predict(X_val)

        # Metrics
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average='weighted')
        recall = recall_score(y_val, y_pred, average='weighted')
        f1 = f1_score(y_val, y_pred, average='weighted')

        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1-Score: {f1:.4f}")

        # Classification report
        report = classification_report(y_val, y_pred, target_names=label_encoder.classes_)
        logger.info("\nClassification Report:\n" + report)

        # Save report
        metrics = {
            'model': 'crime_prediction',
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'n_samples': len(y_val),
            'evaluated_at': datetime.now().isoformat()
        }

        with open(REPORT_DIR / 'crime_evaluation.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"✓ Report saved to {REPORT_DIR / 'crime_evaluation.json'}")

    except Exception as e:
        logger.error(f"✗ Error evaluating crime model: {str(e)}")


def evaluate_audio_model():
    """Evaluate audio model"""
    logger.info("=" * 50)
    logger.info("EVALUATING AUDIO MODEL")
    logger.info("=" * 50)

    try:
        # Load model
        from train import AudioCNN
        with open(MODEL_DIR / 'audio_label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)

        # Load validation data
        X_val = np.load(VALIDATION_DIR / 'audio' / 'audio_features_val.npy')
        y_val = np.load(VALIDATION_DIR / 'audio' / 'audio_labels_val.npy', allow_pickle=True)
        y_val = label_encoder.transform(y_val)

        # Load model state
        input_dim = X_val.shape[1]
        num_classes = len(np.unique(y_val))
        model = AudioCNN(input_dim, num_classes).to(device)
        model.load_state_dict(torch.load(MODEL_DIR / 'audio_model.pth', map_location=device))
        model.eval()

        # Predict
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_val).to(device)
            outputs = model(X_tensor)
            _, y_pred = outputs.max(1)
            y_pred = y_pred.cpu().numpy()

        # Metrics
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average='weighted')
        recall = recall_score(y_val, y_pred, average='weighted')
        f1 = f1_score(y_val, y_pred, average='weighted')

        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1-Score: {f1:.4f}")

        # Save report
        metrics = {
            'model': 'audio_classification',
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'n_samples': len(y_val),
            'evaluated_at': datetime.now().isoformat()
        }

        with open(REPORT_DIR / 'audio_evaluation.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"✓ Report saved to {REPORT_DIR / 'audio_evaluation.json'}")

    except Exception as e:
        logger.error(f"✗ Error evaluating audio model: {str(e)}")


def evaluate_video_model():
    """Evaluate video model"""
    logger.info("=" * 50)
    logger.info("EVALUATING VIDEO MODEL")
    logger.info("=" * 50)

    try:
        # Load model
        from train import Video3DCNN
        with open(MODEL_DIR / 'video_label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)

        # Load validation data
        X_val = np.load(VALIDATION_DIR / 'video' / 'video_frames_val.npy')
        y_val = np.load(VALIDATION_DIR / 'video' / 'video_labels_val.npy', allow_pickle=True)
        y_val = label_encoder.transform(y_val)

        # Load model
        num_classes = len(np.unique(y_val))
        model = Video3DCNN(num_classes).to(device)
        model.load_state_dict(torch.load(MODEL_DIR / 'video_model.pth', map_location=device))
        model.eval()

        # Predict in batches
        batch_size = 4
        all_preds = []

        with torch.no_grad():
            for i in range(0, len(X_val), batch_size):
                batch = X_val[i:i+batch_size]
                X_tensor = torch.FloatTensor(batch).to(device)
                outputs = model(X_tensor)
                _, preds = outputs.max(1)
                all_preds.extend(preds.cpu().numpy())

        y_pred = np.array(all_preds)

        # Metrics
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average='weighted')
        recall = recall_score(y_val, y_pred, average='weighted')
        f1 = f1_score(y_val, y_pred, average='weighted')

        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1-Score: {f1:.4f}")

        # Save report
        metrics = {
            'model': 'video_classification',
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'n_samples': len(y_val),
            'evaluated_at': datetime.now().isoformat()
        }

        with open(REPORT_DIR / 'video_evaluation.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"✓ Report saved to {REPORT_DIR / 'video_evaluation.json'}")

    except Exception as e:
        logger.error(f"✗ Error evaluating video model: {str(e)}")


def evaluate_nlp_model():
    """Evaluate NLP model"""
    logger.info("=" * 50)
    logger.info("EVALUATING NLP MODEL")
    logger.info("=" * 50)

    try:
        # Load model
        from train import TextLSTM
        import json as json_lib

        with open(MODEL_DIR / 'nlp_label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)

        # Load validation data
        val_df = pd.read_csv(VALIDATION_DIR / 'nlp' / 'nlp_val.csv')

        # Load vocabulary
        with open(TRAINING_DIR / 'nlp' / 'vocabulary.json', 'r') as f:
            vocab = json_lib.load(f)

        # Convert to sequences (same as training)
        max_length = 100
        sequences = []
        for text in val_df['text'].values:
            tokens = str(text).split()
            sequence = [vocab.get(token, vocab.get('<UNK>', 0)) for token in tokens]
            if len(sequence) < max_length:
                sequence += [0] * (max_length - len(sequence))
            else:
                sequence = sequence[:max_length]
            sequences.append(sequence)

        X_val = np.array(sequences)
        y_val = label_encoder.transform(val_df['label'].values)

        # Load model
        vocab_size = len(vocab)
        num_classes = len(np.unique(y_val))
        model = TextLSTM(vocab_size, 128, 256, num_classes).to(device)
        model.load_state_dict(torch.load(MODEL_DIR / 'nlp_model.pth', map_location=device))
        model.eval()

        # Predict
        with torch.no_grad():
            X_tensor = torch.LongTensor(X_val).to(device)
            outputs = model(X_tensor)
            _, y_pred = outputs.max(1)
            y_pred = y_pred.cpu().numpy()

        # Metrics
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average='weighted')
        recall = recall_score(y_val, y_pred, average='weighted')
        f1 = f1_score(y_val, y_pred, average='weighted')

        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1-Score: {f1:.4f}")

        # Save report
        metrics = {
            'model': 'nlp_classification',
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'n_samples': len(y_val),
            'evaluated_at': datetime.now().isoformat()
        }

        with open(REPORT_DIR / 'nlp_evaluation.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"✓ Report saved to {REPORT_DIR / 'nlp_evaluation.json'}")

    except Exception as e:
        logger.error(f"✗ Error evaluating NLP model: {str(e)}")


def evaluate_vision_model():
    """Evaluate vision model"""
    logger.info("=" * 50)
    logger.info("EVALUATING VISION MODEL")
    logger.info("=" * 50)

    try:
        # Load model
        from train import VisionCNN
        with open(MODEL_DIR / 'vision_label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)

        # Load validation data
        X_val = np.load(VALIDATION_DIR / 'vision' / 'images_val.npy')
        y_val = np.load(VALIDATION_DIR / 'vision' / 'labels_val.npy', allow_pickle=True)
        y_val = label_encoder.transform(y_val)

        # Load model
        num_classes = len(np.unique(y_val))
        model = VisionCNN(num_classes).to(device)
        model.load_state_dict(torch.load(MODEL_DIR / 'vision_model.pth', map_location=device))
        model.eval()

        # Predict in batches
        batch_size = 32
        all_preds = []

        with torch.no_grad():
            for i in range(0, len(X_val), batch_size):
                batch = X_val[i:i+batch_size]
                X_tensor = torch.FloatTensor(batch).to(device)
                outputs = model(X_tensor)
                _, preds = outputs.max(1)
                all_preds.extend(preds.cpu().numpy())

        y_pred = np.array(all_preds)

        # Metrics
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average='weighted')
        recall = recall_score(y_val, y_pred, average='weighted')
        f1 = f1_score(y_val, y_pred, average='weighted')

        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1-Score: {f1:.4f}")

        # Save report
        metrics = {
            'model': 'vision_classification',
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'n_samples': len(y_val),
            'evaluated_at': datetime.now().isoformat()
        }

        with open(REPORT_DIR / 'vision_evaluation.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"✓ Report saved to {REPORT_DIR / 'vision_evaluation.json'}")

    except Exception as e:
        logger.error(f"✗ Error evaluating vision model: {str(e)}")


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate SmartAiCity models')
    parser.add_argument('--all', action='store_true', help='Evaluate all models')
    parser.add_argument('--crime', action='store_true', help='Evaluate crime model')
    parser.add_argument('--audio', action='store_true', help='Evaluate audio model')
    parser.add_argument('--video', action='store_true', help='Evaluate video model')
    parser.add_argument('--nlp', action='store_true', help='Evaluate NLP model')
    parser.add_argument('--vision', action='store_true', help='Evaluate vision model')

    args = parser.parse_args()

    # If no specific flag, evaluate all
    if not any([args.crime, args.audio, args.video, args.nlp, args.vision]):
        args.all = True

    logger.info("Starting SmartAiCity Model Evaluation")
    logger.info("=" * 50)

    try:
        if args.all or args.crime:
            evaluate_crime_model()

        if args.all or args.audio:
            evaluate_audio_model()

        if args.all or args.video:
            evaluate_video_model()

        if args.all or args.nlp:
            evaluate_nlp_model()

        if args.all or args.vision:
            evaluate_vision_model()

        logger.info("=" * 50)
        logger.info("✓ EVALUATION COMPLETED SUCCESSFULLY")
        logger.info(f"✓ Reports saved to: {REPORT_DIR}")
        logger.info("=" * 50)

        # Print summary
        logger.info("\nEvaluation Summary:")
        for model_type in ['crime', 'audio', 'video', 'nlp', 'vision']:
            report_file = REPORT_DIR / f'{model_type}_evaluation.json'
            if report_file.exists():
                with open(report_file, 'r') as f:
                    metrics = json.load(f)
                logger.info(f"  {model_type.upper()}: "
                          f"Accuracy={metrics['accuracy']:.4f}, "
                          f"F1={metrics['f1_score']:.4f}")

    except Exception as e:
        logger.error(f"✗ Evaluation failed: {str(e)}")
        raise


if __name__ == '__main__':
    # Need TRAINING_DIR for NLP vocabulary
    TRAINING_DIR = BASE_DIR / 'datasets' / 'training'
    main()
