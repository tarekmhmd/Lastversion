from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/vision/', include('vision.urls')),
    path('api/audio/', include('audio.urls')),
    path('api/prediction/', include('prediction.urls')),
    path('api/nlp/', include('nlp.urls')),
    path('api/dashboard/', include('dashboard.urls')),
    path('api/alerts/', include('alerts.urls')),
    path('api/auth/', include('users.urls')),
    path('api/iot/', include('iot.urls')),
    path('api/analytics/', include('analytics.urls')),
    path('api/token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('api/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
