# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from subprocess import Popen, PIPE
from os import devnull
from livestreamer import streams
from django.db import models
from django.contrib.postgres.fields import ArrayField

from nntcpclient import process_image

# Create your models here.
class WatcherManager(models.Manager):
    def create_watcher(self, **kwargs):
        watcher = self.create(**kwargs)
        watcher.set_stream_url()
        return watcher

class Watcher(models.Model):
    youtube_url = models.CharField(max_length=200)
    stream_url = models.CharField(max_length=200)
    sms_number = models.CharField(max_length=12)
    objects_to_watch = ArrayField(models.CharField(max_length=30))
    objects = WatcherManager()

    def set_stream_url(self):
        self.stream_url = streams(self.youtube_url)['medium']

    def get_objects_now(self):
        proc = Popen('ffmpeg -i %s -f image2pipe -vframes 1 -' % self.stream_url,
                     shell=True, stderr=open(devnull, 'w'), stdout=PIPE)
        args = {'query': ':darknet:yolo9000:resize-img:0.24', 'host': '23.96.10.248',
                'port': '6868', 'openimg': proc.stdout, 'show_image': '0'}
        msg = process_image(args)
        print msg
