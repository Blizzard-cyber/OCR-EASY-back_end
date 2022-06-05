from django.urls import path, re_path,include
from . import views
from rest_framework.routers import DefaultRouter

# router = DefaultRouter()
# router.register('ocr_rco', views.rcoViewSet)
#
# urlpatterns = [
#     path('', include(router.urls)),
# ]
urlpatterns = [
    re_path(r'ocr_rco$', views.ocr_rco, name='ocr_rco'),
    re_path(r'show_context$', views.show_context, name='show_context'),
]
