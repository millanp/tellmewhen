os.environ.setdefault("DJANGO_SETTINGS_MODULE", "tellmewhen.settings")

from apscheduler.schedulers.blocking import BlockingScheduler
from watcher.models import Watcher
import os
import redis
from rq import Worker, Queue, Connection

sched = BlockingScheduler()
listen = ['high', 'default', 'low']

redis_url = os.getenv('REDISTOGO_URL', 'redis://localhost:6379')

conn = redis.from_url(redis_url)

with Connection(conn):
    worker = Worker(map(Queue, listen))
    worker.work()


@sched.scheduled_job('interval', minutes=1)
def queue_object_check():
    Queue(connection=conn).enqueue(Watcher.objects.check_for_new_objects)


sched.start()



listen = ['high', 'default', 'low']

redis_url = os.getenv('REDISTOGO_URL', 'redis://localhost:6379')

conn = redis.from_url(redis_url)

with Connection(conn):
    worker = Worker(map(Queue, listen))
    worker.work()
