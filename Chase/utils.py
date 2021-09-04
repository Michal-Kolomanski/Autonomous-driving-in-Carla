import settings
from datetime import datetime


class ColoredPrint:
    def __init__(self):
        self.PINK = '\033[95m'
        self.OKBLUE = '\033[94m'
        self.OKGREEN = '\033[92m'
        self.WARNING = '\033[93m'
        self.FAIL = '\033[91m'
        self.ENDC = '\033[0m'

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
        # reward_num == 4:
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

    # print(f"reward: {reward}, ab_distance: {ab_distance}")

    return reward, done

