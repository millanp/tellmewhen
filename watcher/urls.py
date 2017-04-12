from django.conf.urls import url
import views

urlpatterns = [
    url(r'^watch/(?P<msg>[\w-]+)$', views.StartWatchingView.as_view()),
]
