# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render
from django.views import View
from django.http import HttpResponse
from twilio.rest import Client

# Create your views here.
class StartWatchingView(View):
	def get(self, request, msg=None):
		client = Client()
		message = client.messages.create(
			to='+12533328365', 
			from_='+12532851538',
			body=msg)
		return HttpResponse()

	def post(self, request):
		number = request.POST['cell']
		to_watch = request.POST['watchlist'].split()
		return HttpResponse()