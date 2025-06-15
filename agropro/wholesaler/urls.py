from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path("",views.profile,name='profiles'),
    path("profiles",views.profile,name='profiles'),
    path("farmersmarket/",views.market,name='market'),
    path("notifications/",views.notification,name='notification'),
    path("editProfile",views.editProfile,name='editProfile'),
     path('blogs', views.blog_Wholesale, name='blogs'),
]