from django.conf.urls import url
import views

urlpatterns = [
    url(ur'^watch/(?P<msg>.*)$', views.StartWatchingView.as_view()),
]
