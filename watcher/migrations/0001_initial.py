# -*- coding: utf-8 -*-
# Generated by Django 1.11 on 2017-04-12 23:05
from __future__ import unicode_literals

import django.contrib.postgres.fields
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Watcher',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('youtube_url', models.CharField(max_length=200)),
                ('stream_url', models.CharField(max_length=200)),
                ('sms_number', models.CharField(max_length=12)),
                ('objects_to_watch', django.contrib.postgres.fields.ArrayField(base_field=models.CharField(max_length=30), size=None)),
            ],
        ),
    ]
