import glob
import os
import random
import sys
import time
import cv2
import math
import torch

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import numpy as np

actor_list = []

IMAGE_WIDTH = 480
IMAGE_HEIGHT = 480
SHOW_PREVIEW = False
EPISODE_TIME = 10


class CarlaEnv:
    SHOW_CAMERA = SHOW_PREVIEW
    STEER_AMOUNT = 1.0
    image_width = IMAGE_WIDTH
    image_height = IMAGE_HEIGHT
    front_camera = None

    def __init__(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(60.0)
        #self.client.load_world('Town01')
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.my_vehicle = self.blueprint_library.filter("model3")[0]
        self.image_width = IMAGE_WIDTH
        self.image_height = IMAGE_HEIGHT

    def reset(self):
        self.collsion_history = []
        self.actor_list = []


        self.spawn_point = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.my_vehicle, self.spawn_point)
        self.actor_list.append(self.vehicle)

        self.camera_blueprint = self.blueprint_library.find('sensor.camera.rgb')
        self.camera_blueprint.set_attribute("image_size_x", f"{self.image_width}")
        self.camera_blueprint.set_attribute("image_size_y", f"{self.image_height}")
        self.camera_blueprint.set_attribute("fov", f"110")

        spawn_sensor = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.camera_sensor = self.world.spawn_actor(self.camera_blueprint, spawn_sensor, attach_to=self.vehicle)
        self.actor_list.append(self.camera_sensor)
        self.camera_sensor.listen(lambda data: self.processing_image(data))

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(4)

        self.collision_sensor_blueprint = self.blueprint_library.find('sensor.other.collision')
        self.collsion_sensor = self.world.spawn_actor(self.collision_sensor_blueprint, spawn_sensor,
                                                      attach_to=self.vehicle)
        self.actor_list.append(self.collsion_sensor)
        self.collsion_sensor.listen(lambda event: self.collision_data(event))

        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        self.actor_list.clear()
        return self.front_camera

    def collision_data(self, event):
        self.collsion_history.append(event)

    def processing_image(self, image):
        img = np.array(image.raw_data)
        img = img.reshape((self.image_height, self.image_width, 4))
        img = img[:, :, :3]
        if self.SHOW_CAMERA:
            cv2.imshow("", img)
            cv2.waitKey(1)
        self.front_camera = (torch.Tensor(img).view(3, 480, 480)).numpy()

    def step(self, action):
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, brake=-1 * self.STEER_AMOUNT))
        elif action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, brake=0))
        elif action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, brake=1 * self.STEER_AMOUNT))

        velocity = self.vehicle.get_velocity()
        speed_kmh = int(3.8 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2))

        if len(self.collsion_history) != 0:
            done = True
            reward = -200
        elif speed_kmh < 30:
            done = False
            reward = -1
        else:
            done = False
            reward = 1

        if self.episode_start + EPISODE_TIME < time.time():
            done = True

        return self.front_camera, reward, done, None


