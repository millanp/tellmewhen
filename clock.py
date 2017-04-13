from apscheduler.schedulers.blocking import BlockingScheduler
from worker import conn
from watcher.models import Watcher
from rq import Queue

sched = BlockingScheduler()


@sched.scheduled_job('interval', minutes=1)
def queue_object_check():
    Queue(connection=conn).enqueue(Watcher.objects.check_for_new_objects)

sched.start()
