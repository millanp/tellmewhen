# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render
from django.views import View
from django.http import HttpResponse
from twilio.rest import Client
from models import Watcher

# Create your views here.


class StartWatchingView(View):
    """
    A view that receives HTTP requests
    to start watching a youtube livestream
    for given objects, and texts the
    given SMS number when an object appears.
    """

    def get(self, request, msg=None):
        client = Client()
        client.messages.create(
            to='+12533328365',
            from_='+12532851538',
            body=msg)
        return HttpResponse()

    def post(self, request):
        number = request.POST['cell']
        url = request.POST['url']
        to_watch = request.POST['watchlist'].split()
        Watcher.objects.create_watcher(
            youtube_url=url, objects_to_watch=to_watch, sms_number=number)
        return HttpResponse()
