from django.urls import path
from predict import views

urlpatterns = [
    path('', views.indexView, name="home"),
    path('prediction', views.predictionView, name="predictionView"),
]
