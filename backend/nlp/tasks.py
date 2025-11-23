from celery import shared_task
from .models import CitizenReport
import logging
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from django.conf import settings
import os

logger = logging.getLogger(__name__)

# Global model cache
_sentiment_analyzer = None
_text_classifier = None

def get_sentiment_analyzer():
    """
    Initialize sentiment analysis pipeline
    Uses multilingual model that supports Arabic
    """
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        try:
            # Use multilingual sentiment model
            _sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="nlptown/bert-base-multilingual-uncased-sentiment",
                truncation=True
            )
            logger.info("Sentiment analyzer loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load sentiment analyzer: {str(e)}")
            raise
    return _sentiment_analyzer

def classify_report_text(text):
    """
    Classify Arabic text into report categories
    Uses keyword-based classification with sentiment analysis
    """
    try:
        # Get sentiment
        sentiment_analyzer = get_sentiment_analyzer()
        sentiment_result = sentiment_analyzer(text[:512])[0]  # Limit to 512 tokens

        # Extract sentiment score (1-5 stars)
        sentiment_label = sentiment_result['label']
        sentiment_score = sentiment_result['score']

        # Convert star rating to numeric
        stars = int(sentiment_label.split()[0])  # Extract number from "X stars"

        # Keyword-based classification for Arabic reports
        text_lower = text.lower()

        # Define keywords for each category
        crime_keywords = ['سرقة', 'جريمة', 'قتل', 'اعتداء', 'سطو', 'نصب', 'احتيال']
        accident_keywords = ['حادث', 'تصادم', 'إصابة', 'حريق', 'انفجار']
        emergency_keywords = ['طوارئ', 'مستعجل', 'خطر', 'نجدة', 'إسعاف']
        traffic_keywords = ['مرور', 'ازدحام', 'طريق', 'إشارة', 'توقف']
        complaint_keywords = ['شكوى', 'مشكلة', 'تذمر', 'اعتراض']
        suggestion_keywords = ['اقتراح', 'فكرة', 'تحسين', 'تطوير']

        # Check keywords
        if any(keyword in text_lower for keyword in crime_keywords):
            report_type = 'crime'
            confidence = 0.85
        elif any(keyword in text_lower for keyword in accident_keywords):
            report_type = 'accident'
            confidence = 0.82
        elif any(keyword in text_lower for keyword in emergency_keywords):
            report_type = 'emergency'
            confidence = 0.90
        elif any(keyword in text_lower for keyword in traffic_keywords):
            report_type = 'traffic'
            confidence = 0.75
        elif any(keyword in text_lower for keyword in complaint_keywords):
            report_type = 'complaint'
            confidence = 0.70
        elif any(keyword in text_lower for keyword in suggestion_keywords):
            report_type = 'suggestion'
            confidence = 0.72
        else:
            report_type = 'other'
            confidence = 0.50

        # Adjust confidence based on sentiment score
        if stars <= 2:  # Negative sentiment increases urgency
            if report_type in ['crime', 'accident', 'emergency']:
                confidence = min(confidence + 0.1, 0.95)

        return report_type, confidence, stars, sentiment_score

    except Exception as e:
        logger.error(f"Error in text classification: {str(e)}")
        return 'other', 0.5, 3, 0.5

@shared_task
def process_arabic_report(report_id):
    """
    Process Arabic citizen reports using NLP
    """
    try:
        report = CitizenReport.objects.get(id=report_id)
        logger.info(f"Starting report processing: {report_id}")

        # Classify the report
        report_type, confidence, sentiment_stars, sentiment_score = classify_report_text(report.report_text)

        # Determine urgency level
        urgent_types = ['crime', 'accident', 'emergency']
        if report_type in urgent_types:
            urgency_level = 'high'
        elif report_type == 'traffic':
            urgency_level = 'medium'
        else:
            urgency_level = 'low'

        # Update with low sentiment (negative feedback)
        if sentiment_stars <= 2:
            urgency_level = 'high' if urgency_level != 'high' else urgency_level

        # Update report
        report.report_type = report_type
        report.confidence = confidence
        report.urgency_level = urgency_level
        report.is_processed = True
        report.save()

        logger.info(f"Report processed - Type: {report_type}, Confidence: {confidence:.2f}, Urgency: {urgency_level}")
        return f"Report {report_id} processed - Type: {report_type}"

    except CitizenReport.DoesNotExist:
        logger.error(f"Report not found: {report_id}")
        return "Report not found"
    except Exception as e:
        logger.error(f"Error processing report: {str(e)}", exc_info=True)
        return f"Error: {str(e)}"

@shared_task
def batch_process_reports():
    """
    Batch process unprocessed citizen reports
    """
    unprocessed_reports = CitizenReport.objects.filter(is_processed=False)
    count = unprocessed_reports.count()

    for report in unprocessed_reports:
        process_arabic_report.delay(report.id)

    logger.info(f"Queued {count} reports for processing")
    return f"Queued {count} reports for processing"

@shared_task
def extract_entities_from_report(report_id):
    """
    Extract named entities from report (locations, persons, dates)
    """
    try:
        report = CitizenReport.objects.get(id=report_id)
        text = report.report_text

        # Simple entity extraction using keywords and patterns
        # In production, use CAMeL Tools or Farasa for Arabic NER
        entities = {
            'locations': [],
            'persons': [],
            'dates': [],
            'times': []
        }

        # Simple location extraction (cities in Egypt)
        locations = ['القاهرة', 'الإسكندرية', 'المنصورة', 'طنطا', 'الزقازيق', 'دمياط', 'بورسعيد']
        for loc in locations:
            if loc in text:
                entities['locations'].append(loc)

        # Simple time expressions
        time_keywords = ['صباح', 'مساء', 'ظهر', 'ليل', 'الساعة']
        for keyword in time_keywords:
            if keyword in text:
                entities['times'].append(keyword)

        report.entities = entities
        report.save()

        logger.info(f"Extracted entities from report {report_id}")
        return f"Entities extracted from report {report_id}"

    except CitizenReport.DoesNotExist:
        logger.error(f"Report not found: {report_id}")
        return f"Report not found: {report_id}"
    except Exception as e:
        logger.error(f"Error extracting entities: {str(e)}")
        return f"Error: {str(e)}"

@shared_task
def analyze_sentiment_trends():
    """
    Analyze sentiment trends across all reports
    """
    try:
        recent_reports = CitizenReport.objects.filter(is_processed=True).order_by('-created_at')[:100]

        positive_count = recent_reports.filter(report_type='suggestion').count()
        negative_count = recent_reports.filter(report_type__in=['crime', 'complaint', 'accident']).count()

        trend = {
            'total': recent_reports.count(),
            'positive': positive_count,
            'negative': negative_count,
            'neutral': recent_reports.count() - positive_count - negative_count
        }

        logger.info(f"Sentiment trend analysis complete: {trend}")
        return trend

    except Exception as e:
        logger.error(f"Error analyzing sentiment trends: {str(e)}")
        return {"error": str(e)}
