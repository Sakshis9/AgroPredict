from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path("",views.profile,name='profile'),
    path("profile",views.profile,name='profile'),
    path("notifications/",views.notification,name='notification'),
    path("prediction",views.prediction,name='prediction'),
    path('predict_soil', views.predict_soil, name='predict_soil'),
    path('blog', views.blog_list, name='blog'),
    path("editProfile",views.editProfile,name='editProfile'),
    path("setCrop",views.setCrop,name='setCrop'),
    path("removeCrop/<int:crop_id>",views.removeCrop,name='removeCrop')
]