from rest_framework import viewsets, permissions
from rest_framework.decorators import action
from rest_framework.response import Response
from django.utils import timezone
from .models import CrimePrediction, HotZone
from .serializers import CrimePredictionSerializer, HotZoneSerializer

class CrimePredictionViewSet(viewsets.ModelViewSet):
    queryset = CrimePrediction.objects.all()
    serializer_class = CrimePredictionSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    @action(detail=False, methods=['get'])
    def today(self, request):
        today_predictions = self.queryset.filter(predicted_date=timezone.now().date())
        serializer = self.get_serializer(today_predictions, many=True)
        return Response(serializer.data)
    
    @action(detail=False, methods=['get'])
    def high_probability(self, request):
        high_prob = self.queryset.filter(probability__gte=0.7)
        serializer = self.get_serializer(high_prob, many=True)
        return Response(serializer.data)

class HotZoneViewSet(viewsets.ModelViewSet):
    queryset = HotZone.objects.all()
    serializer_class = HotZoneSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    @action(detail=False, methods=['get'])
    def critical(self, request):
        critical_zones = self.queryset.filter(risk_level='critical')
        serializer = self.get_serializer(critical_zones, many=True)
        return Response(serializer.data)
