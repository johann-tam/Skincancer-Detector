"""
Original author(s): Auto generated
Modified by:

File purpose: Store urls used in the system
"""

from django.urls import path
from . import views
from django.contrib.auth import views as auth_views

urlpatterns = [
    path('', views.make_prediction, name='home'),
    path('datasets/', views.data_management, name='datasets'),
    path('models/', views.model_management, name='models'),
    path('logout/', auth_views.LogoutView.as_view(), name='logout',),
    path('register/', views.register_user, name='register'),
    path('logout_user', views.logout_user, name='logout_user'),
    path('login_user', views.login_user, name='login_user'),
    path('history', views.show_history, name='history'),
]
