from rest_framework import viewsets, permissions, status
from rest_framework.decorators import action
from rest_framework.response import Response
from .models import CitizenReport
from .serializers import CitizenReportSerializer, CitizenReportCreateSerializer
from .tasks import process_arabic_report

class CitizenReportViewSet(viewsets.ModelViewSet):
    queryset = CitizenReport.objects.all()
    serializer_class = CitizenReportSerializer
    permission_classes = [permissions.AllowAny]
    
    def get_serializer_class(self):
        if self.action == 'create':
            return CitizenReportCreateSerializer
        return CitizenReportSerializer
    
    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        report = serializer.save()
        
        # Process the report asynchronously
        process_arabic_report.delay(report.id)
        
        return Response(
            CitizenReportSerializer(report).data,
            status=status.HTTP_201_CREATED
        )
    
    @action(detail=False, methods=['get'])
    def unprocessed(self, request):
        unprocessed = self.queryset.filter(is_processed=False)
        serializer = self.get_serializer(unprocessed, many=True)
        return Response(serializer.data)
    
    @action(detail=False, methods=['get'])
    def urgent(self, request):
        urgent = self.queryset.filter(urgency_level__in=['high', 'critical'])
        serializer = self.get_serializer(urgent, many=True)
        return Response(serializer.data)
