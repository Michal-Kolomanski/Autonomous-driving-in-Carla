import settings
import time
from datetime import datetime


class Timer:
    def __init__(self):
        self.time_started = None
        self.time_paused = None
        self.paused = False

    def start(self):
        """ Starts an internal timer """
        self.time_started = time.time()

    def pause(self):
        """ Pauses the timer """
        if self.time_started is None:
            raise ValueError("Timer not started")
        if self.paused:
            raise ValueError("Timer is already paused")
        self.time_paused = time.time()
        self.paused = True

    def resume(self):
        """ Resumes the timer by adding the pause time to the start time """
        if self.time_started is None:
            raise ValueError("Timer not started")
        if not self.paused:
            raise ValueError("Timer is not paused")
        pause_time = time.time() - self.time_paused
        self.time_started = self.time_started + pause_time
        self.paused = False

    def get(self):
        """ Get the time """
        if self.time_started is None:
            raise ValueError("Timer not started")
        if self.paused:
            return round(self.time_paused - self.time_started, 3)
        else:
            return round(time.time() - self.time_started, 3)


class ColoredPrint:
    """
    Creates colorful logs
    """
    def __init__(self):
        self.PINK = '\033[95m'
        self.OKBLUE = '\033[94m'
        self.OKGREEN = '\033[92m'
        self.WARNING = '\033[93m'
        self.FAIL = '\033[91m'
        self.ENDC = '\033[0m'
        self.msg = None

    def disable(self):
        self.PINK = ''
        self.OKBLUE = ''
        self.OKGREEN = ''
        self.WARNING = ''
        self.FAIL = ''
        self.ENDC = ''

    def store(self):
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open('logfile.log', mode='a') as file_:
            file_.write(f"{self.msg} -- {date}")
            file_.write("\n")

    def success(self, *args, **kwargs):
        self.msg = ' '.join(map(str, args))
        print(self.OKGREEN + self.msg + self.ENDC, **kwargs)
        return self

    def info(self, *args, **kwargs):
        self.msg = ' '.join(map(str, args))
        print(self.OKBLUE + self.msg + self.ENDC, **kwargs)
        return self

    def warn(self, *args, **kwargs):
        self.msg = ' '.join(map(str, args))
        print(self.WARNING + self.msg + self.ENDC, **kwargs)
        return self

    def err(self, *args, **kwargs):
        self.msg = ' '.join(map(str, args))
        print(self.FAIL + self.msg + self.ENDC, **kwargs)
        return self

    def pink(self, *args, **kwargs):
        self.msg = ' '.join(map(str, args))
        print(self.PINK + self.msg + self.ENDC, **kwargs)
        return self


def reward_function(collision_history_list, ab_distance, timer):
    """
    Distance reward
    """
    reward_num = settings.REWARD_NUMBER

    if reward_num == 1:
        col_reward = -1
        timer_reward = round(1/14 * timer, 3)

        if 5 <= ab_distance <= 25:
            route_distance_reward = 1
        else:
            route_distance_reward = 0

    elif reward_num == 2:
        col_reward = -10
        timer_reward = round(1/11 * timer, 3)

        if 5 <= ab_distance <= 25:
            route_distance_reward = 1
        else:
            route_distance_reward = -1

    elif reward_num == 3:
        col_reward = -10
        timer_reward = round(1/3 * timer, 3)

        route_distance_reward = round(-1/6 * ab_distance + 4, 3)

    else:
        col_reward = -1
        timer_reward = round(1/12 * timer, 3)

        route_distance_reward = round(-0.038 * ab_distance + 1, 3)

    if len(collision_history_list) != 0:
        done = True  # There was a collision end the episode
    else:
        done = False
        col_reward = 0

    if ab_distance >= 60:
        done = True

    reward = round(col_reward + route_distance_reward + timer_reward, 3)

    return reward, done

