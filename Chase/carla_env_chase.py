import glob
import os
import random
import sys
import time
import math
import torch
import pickle
import pygame
import threading

import settings
from utils import ColoredPrint, reward_function
from ACTIONS import ACTIONS as ac

try:
    sys.path.append(glob.glob(settings.CARLA_EGG_PATH % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from carla import ColorConverter as cc
import numpy as np
import subprocess
import cv2

# Global settings
fps = settings.FPS
show_cam = settings.SHOW_CAM
sleep_time = settings.SLEEP_BETWEEN_ACTIONS


def start_carla_server(args):
    return subprocess.Popen(f'CarlaUE4.exe ' + args, cwd=settings.CARLA_PATH, shell=True)


class CarlaEnv:
    """
    Create Carla environment
    """
    def __init__(self, scenario, action_space='discrete',  camera='rgb', res_x=80, res_y=80, port=2000, manual_control=False):
        # Run the server on 127.0.0.1/port
        start_carla_server(f'-windowed -carla-server -fps={fps} -ResX=640 -ResY=480 -quality-level=Low'
                           f' -carla-world-port={port}')
        self.client = carla.Client("localhost", port)
        self.client.set_timeout(10.0)

        # Enable to use colors
        self.log = ColoredPrint()

        # Make sure that server and client versions are the same
        client_ver = self.client.get_client_version()
        server_ver = self.client.get_server_version()

        if client_ver == server_ver:
            self.log.success(f"Client version: {client_ver}, Server version: {server_ver}")
        else:
            self.log.warn(f"Client version: {client_ver}, Server version: {server_ver}")

        self.client.load_world('Town03')
        self.world = self.client.get_world()
        self.settings = self.world.get_settings()
        self.clock = pygame.time.Clock()
        self.camera_type = camera

        # List of available agents with attributes
        self.blueprint_library = self.world.get_blueprint_library()
        self.map = self.world.get_map()

        self.scenario_list = scenario
        try:
            self.scenario = self.scenario_list[0]  # Single scenario
        except:
            self.scenario = False

        """
        a - vehicle which is chasing
        b - vehicle which is being chased
        sp - spawn point
        """
        self.a_sp, self.b_sp, self.ride_history = self.create_scenario()
        self.a_sp_loc, self.b_sp_loc = self.a_sp.location, self.b_sp.location

        # Set the spectator
        self.spectator = self.set_spectator()
        # Environment possible actions
        self.action_space = self.create_action_space(action_space)
        # List of all actors in the environment
        self.actor_list = []

        self.transform = carla.Transform(carla.Location(x=2.5, z=0.7))

        self.manual_control = manual_control

        self.b_vehicle = self.spawn_car('carlacola', self.b_sp)  # Chased car
        self.prev_b_vehicle_loc = 0  # Previous location of the chased car. Used for drawing a route
        if not manual_control:
            self.a_vehicle = self.spawn_car('model3', self.a_sp)  # Car which is chasing

        # Manages the basic movement of a vehicle using typical driving controls
        self.control = carla.VehicleControl()

        # Images X, Y resolutions
        self.res_x = res_x
        self.res_y = res_y

        # Variables which have to reset at the end of each episode
        self.collision_history_list = []

        # The number of steps in one episode
        self.step_counter = 0

        # Cameras
        self.show_cam = show_cam
        self.front_camera = None

        # Vehicle B movement thread
        self.thread = None
        self.done = False
        self.timer = None

    def create_scenario(self):
        """
        Create a scenario based on integer, input value
        """
        if self.scenario == 1:
            # Short straight
            ride_file = 'drives/ride1.p'
            ride_history, b_sp_loc, b_sp_rot = self.load_ride(ride_file)
            b_sp_loc.z += 0.01  # without that there is a collision with the map
            b_sp = carla.Transform(b_sp_loc, b_sp_rot)

            a_sp_loc = carla.Location(b_sp_loc.x, b_sp_loc.y, b_sp_loc.z)
            a_sp_loc.x -= 10  # Spawn behind the chased car
            a_sp = carla.Transform(a_sp_loc, b_sp_rot)

            for i in range(len(ride_history) - 300):  # Make the ride shorter
                ride_history.pop()

            return a_sp, b_sp, ride_history

        elif self.scenario == 2:
            # Long straight
            ride_file = 'drives/ride1.p'
            ride_history, b_sp_loc, b_sp_rot = self.load_ride(ride_file)
            b_sp_loc.z += 0.01  # without that there is a collision with the map
            b_sp = carla.Transform(b_sp_loc, b_sp_rot)

            a_sp_loc = carla.Location(b_sp_loc.x, b_sp_loc.y, b_sp_loc.z)
            a_sp_loc.x -= 10  # Spawn behind the chased car
            a_sp = carla.Transform(a_sp_loc, b_sp_rot)

            return a_sp, b_sp, ride_history

        elif self.scenario == 3:
            # Turn left
            ride_file = 'drives/ride7.p'
            ride_history, b_sp_loc, b_sp_rot = self.load_ride(ride_file)
            b_sp_loc.z += 0.01  # without that there is a collision with the map

            b_sp = carla.Transform(b_sp_loc, b_sp_rot)

            a_sp_loc = carla.Location(b_sp_loc.x, b_sp_loc.y, b_sp_loc.z)
            a_sp_loc.x += 10  # Spawn behind the chased car
            a_sp = carla.Transform(a_sp_loc, b_sp_rot)
            for i in range(len(ride_history) - 250):  # Make the ride shorter
                ride_history.pop()

            return a_sp, b_sp, ride_history

        elif self.scenario == 4:
            # Slow turn left
            ride_file = 'drives/ride9.p'
            ride_history, b_sp_loc, b_sp_rot = self.load_ride(ride_file)
            b_sp_loc.z += 0.01  # without that there is a collision with the map

            b_sp = carla.Transform(b_sp_loc, b_sp_rot)

            a_sp_loc = carla.Location(b_sp_loc.x, b_sp_loc.y, b_sp_loc.z)
            a_sp_loc.x += 10  # Spawn behind the chased car
            a_sp = carla.Transform(a_sp_loc, b_sp_rot)
            for i in range(len(ride_history) - 300):  # Make the ride shorter
                ride_history.pop()

            return a_sp, b_sp, ride_history

        elif self.scenario == 5:
            # Turn right
            ride_file = 'drives/ride14.p'
            ride_history, b_sp_loc, b_sp_rot = self.load_ride(ride_file)
            b_sp_loc.z += 0.01  # without that there is a collision with the map

            b_sp = carla.Transform(b_sp_loc, b_sp_rot)

            a_sp_loc = carla.Location(b_sp_loc.x, b_sp_loc.y, b_sp_loc.z)
            a_sp_loc.x += 10  # Spawn behind the chased car
            a_sp = carla.Transform(a_sp_loc, b_sp_rot)
            for i in range(len(ride_history) - 350):  # Make the ride shorter
                ride_history.pop()

            return a_sp, b_sp, ride_history

        else:
            self.log.err(f"Invalid params: scenario: {self.scenario}")

    @staticmethod
    def load_ride(filepath):
        """
        Load the history of a ride
        :param filepath: pickle file
        :return: full ride hisotry and a spawn point for a chased vehicle
        """
        ride_history = pickle.load(open(filepath, 'rb'))

        sp = ride_history[0]
        b_sp_loc = carla.Location(sp[0], sp[1], sp[2])
        b_sp_rot = carla.Rotation(sp[3], sp[4], sp[5])

        return ride_history, b_sp_loc, b_sp_rot

    def set_spectator(self):
        """
        Get specator's camera angles
        :param d: constant
        :return: self.spectator - spectator's exact location and its angles
        """
        spectator_coordinates = carla.Location(self.a_sp_loc.x,
                                               self.a_sp_loc.y,
                                               self.a_sp_loc.z)
        rotation = carla.Rotation(0, 0, 0)

        if self.scenario in [1, 2]:
            spectator_coordinates.x -= 3
            spectator_coordinates.z += 30
            rotation = carla.Rotation(yaw=0, pitch=-50, roll=0)

        elif self.scenario in [3, 4]:
            spectator_coordinates.x -= 3
            spectator_coordinates.z += 45
            rotation = carla.Rotation(yaw=-210, pitch=-70, roll=0)

        elif self.scenario == 5:
            spectator_coordinates.x -= 3
            spectator_coordinates.z += 45
            rotation = carla.Rotation(yaw=-140, pitch=-70, roll=0)

        self.spectator = self.world.get_spectator()
        """
        yaw - rotating your vision in 2D (left <-, right ->)
        pitch - looking more to the sky or the road 
        roll - leaning your vision (e.g. from | to ->)
        """
        self.spectator.set_transform(carla.Transform(spectator_coordinates, rotation))

        return self.spectator

    def spawn_car(self, model_name, spawn_point):
        """
        Spawn a car
        :return: vehicle
        """
        bp = self.blueprint_library.filter(model_name)[0]
        vehicle = self.world.try_spawn_actor(bp, spawn_point)
        self.actor_list.append(vehicle)

        return vehicle

    def create_action_space(self, action_space):
        """
        Create an action space for an agent
        :param action_space: discrete or continuous
        :return: possible actions to take for an agent
        """
        if action_space == 'discrete':
            self.action_space = [getattr(ac, action) for action in settings.ACTIONS]
            return self.action_space
        else:  # Continuous
            self.action_space = action_space
            return self.action_space

    def add_rgb_camera(self, vehicle):
        """
        Attach RGB camera to the vehicle
        The "RGB" camera acts as a regular camera capturing images from the scene
        """
        rgb_cam_bp = self.blueprint_library.find("sensor.camera.rgb")
        rgb_cam_bp.set_attribute("image_size_x", f"{self.res_x}")
        rgb_cam_bp.set_attribute("image_size_y", f"{self.res_y}")
        rgb_cam_bp.set_attribute("fov", f"110")

        rgb_cam = self.world.spawn_actor(rgb_cam_bp, self.transform, attach_to=vehicle)
        self.actor_list.append(rgb_cam)
        rgb_cam.listen(lambda data: self.process_rgb_img(data))

    def process_rgb_img(self, image):
        """
        Process RGB images
        :param image: raw data from the rgb camera
        :return:
        """
        i = np.array(image.raw_data)
        # Also returns alfa values - not only rgb
        i2 = i.reshape((self.res_y, self.res_x, 4))
        i3 = i2[:, :, :3]

        if self.show_cam:
            # noinspection PyUnresolvedReferences
            cv2.imshow("", i3)
            # noinspection PyUnresolvedReferences
            cv2.waitKey(1)

        # if self.step_counter % 10 == 0:
        #     # noinspection PyUnresolvedReferences
        #     cv2.imwrite('C:\mkoloman\Magisterka\Chase\images\image{}.png'.format(self.step_counter), i3)

        self.front_camera = torch.Tensor(i3).view(3, self.res_x, self.res_y).unsqueeze(0)

    def add_semantic_camera(self, vehicle):
        """
        Attach semantic camera to the vehicle
        The "Semantic Segmentation" camera classifies every object in the view by displaying it in a different color
        according to the object class. E.g. pedestrians appear in a different color than vehicles.
        Original images are totally black
        """
        semantic_cam_bp = self.blueprint_library.find('sensor.camera.semantic_segmentation')
        semantic_cam_bp.set_attribute('image_size_x', f'{self.res_x}')
        semantic_cam_bp.set_attribute('image_size_y', f'{self.res_y}')
        semantic_cam_bp.set_attribute('fov', '110')

        semantic_cam_sensor = self.world.spawn_actor(semantic_cam_bp, self.transform, attach_to=vehicle)

        semantic_cam_sensor.listen(lambda data: self.process_semantic_img(data))
        self.actor_list.append(semantic_cam_sensor)

    def process_semantic_img(self, image):
        """
        Process semantic images
        :param image: raw data from the semantic camera
        """
        image.convert(cc.CityScapesPalette)
        image = np.array(image.raw_data)
        image = image.reshape((self.res_x, self.res_y, 4))
        image = image[:, :, :3]
        if self.show_cam:
            # noinspection PyUnresolvedReferences
            cv2.imshow("", image)
            # noinspection PyUnresolvedReferences
            cv2.waitKey(1)

        # if self.step_counter % 10 == 0:
        #     # noinspection PyUnresolvedReferences
        #     cv2.imwrite('C:\mkoloman\Magisterka\Chase\images\image{}.png'.format(self.step_counter), image)

        self.front_camera = torch.Tensor(image).view(3, self.res_x, self.res_y).unsqueeze(0).float()

    def add_collision_sensor(self, vehicle):
        """
        This sensor, when attached to an actor, it registers an event each time the actor collisions against sth
        in the world. This sensor does not have any configurable attribute.
        """
        col_sensor_bp = self.blueprint_library.find('sensor.other.collision')
        col_sensor_bp = self.world.spawn_actor(col_sensor_bp, self.transform, attach_to=vehicle)
        col_sensor_bp.listen(lambda data: self.collision_data_registering(data))
        self.actor_list.append(col_sensor_bp)

    def collision_data_registering(self, event):
        """
        Register collisions
        :param event: data from the collision sensor
        """
        coll_type = event.other_actor.type_id
        # self.log.err(f"Collision with: {coll_type}")
        self.collision_history_list.append(event)

    def car_control_continuous(self, action, vehicle):
        """
        Manages the basic movement of a vehicle using typical driving controls.
        Instance variables:
        throttle (float) - A scalar value to control the vehicle throttle [0.0, 1.0]. Default is 0.0.
        steer (float) - A scalar value to control the vehicle steering [-1.0, 1.0]. Default is 0.0.
        brake (float) - A scalar value to control the vehicle brake [0.0, 1.0]. Default is 0.0.
        hand_brake (bool) - Determines whether hand brake will be used. Default is False.
        reverse (bool) - Determines whether the vehicle will move backwards. Default is False.
        manual_gear_shift (bool) -Determines whether the vehicle will be controlled by changing gears manually. Default is False.
        gear (int) - States which gear is the vehicle running on.
        """
        gas_value = float(np.clip(action[0], 0, 1))
        brake = float(np.abs(np.clip(action[0], -1, 0)))
        steer = float(np.clip(action[1], -1, 1))
        self.control.throttle = gas_value
        self.control.steer = steer
        self.control.brake = brake
        self.control.hand_brake = False
        self.control.reverse = False
        self.control.manual_gear_shift = False
        vehicle.apply_control(self.control)

    def car_control_discrete(self, action, vehicle):
        self.control.throttle = ac.ACTION_CONTROL[self.action_space[action]][0]
        self.control.brake = ac.ACTION_CONTROL[self.action_space[action]][1]
        self.control.steer = ac.ACTION_CONTROL[self.action_space[action]][2]
        self.control.hand_brake = False
        self.control.reverse = False
        self.control.manual_gear_shift = False
        vehicle.apply_control(self.control)

    @staticmethod
    def calculate_distance(a_location, b_location):
        """
        Calculate distance between two locations in the environment based on the coordinates
        :param a_location:   Carla Location class
        :param b_location:   Carla Location class
        :return: distance    float
        """

        distance = math.sqrt((b_location.x - a_location.x) ** 2 +
                             (b_location.y - a_location.y) ** 2 +
                             (b_location.z - a_location.z) ** 2)

        return distance

    @staticmethod
    def calculate_angle(a_vehicle, b_vehicle):
        """
        Calculate the angle between two actors
        :param a_vehicle: (carla.Actor)
        :param b_vehicle: (carla.Actor)
        :return: angle between two actors
        """
        a_rotation = a_vehicle.get_transform().rotation
        b_rotation = b_vehicle.get_transform().rotation
        angle = a_rotation.yaw - b_rotation.yaw

        return abs(angle)

    def draw_movement(self, vehicle):
        """
        Creates a mark after the car's movement (green X)
        :param vehicle: (carla.Actor)
        return: location of the vehicle
        """
        vehicle_location = vehicle.get_location()
        green = carla.Color(0, 255, 0)
        red = carla.Color(255, 0, 0)

        if vehicle.type_id == "vehicle.tesla.model3":  # Chasing car
            self.world.debug.draw_string(location=vehicle_location, text="X", color=green, life_time=100)
        elif self.prev_b_vehicle_loc:  # Chased car. if it is not the first step
            self.world.debug.draw_line(begin=self.prev_b_vehicle_loc, end=vehicle_location, thickness=0.3, color=red,
                                       life_time=100)

        return vehicle_location

    def chased_vehicle_movement(self):
        """
        Teleports a chased vehicle to the next location from the file
        """
        for frame in self.ride_history:
            if not self.done:
                time.sleep(0.07)
                self.draw_movement(self.b_vehicle)
                self.prev_b_vehicle_loc = self.b_vehicle.get_transform().location
                new_loc = carla.Location(frame[0], frame[1], frame[2])
                new_rotation = carla.Rotation(frame[3], frame[4], frame[5])

                new_point = carla.Transform(new_loc, new_rotation)
                self.b_vehicle.set_transform(new_point)

    def reload_world(self):
        """
        Rest variables at the end of each episode
        """

        self.destroy_agents()
        self.actor_list = []

        if not self.manual_control:
            self.world = self.client.reload_world()

        self.collision_history_list = []
        self.prev_b_vehicle_loc = None

        # The number of steps in one episode
        self.step_counter = 0
        self.done = False

        # Cameras
        self.front_camera = None

    def reset(self, vehicle_for_mc=None):
        """
        Rest environment at the end of each episode
        :return:
        """

        if self.manual_control:
            self.a_vehicle = vehicle_for_mc

        if self.step_counter > 0:  # Omit the first iteration
            self.thread.join()

        self.reload_world()

        self.scenario = random.choice(self.scenario_list)
        self.a_sp, self.b_sp, self.ride_history = self.create_scenario()
        self.a_sp_loc, self.b_sp_loc = self.a_sp.location, self.b_sp.location

        # Set the spectator
        self.spectator = self.set_spectator()

        self.b_vehicle = self.spawn_car('carlacola', self.b_sp)  # Chased car
        if not self.manual_control:
            self.a_vehicle = self.spawn_car('model3', self.a_sp)  # Car which is chasing

        if self.camera_type == 'rgb':
            self.add_rgb_camera(self.a_vehicle)
        elif self.camera_type == 'semantic':
            self.add_semantic_camera(self.a_vehicle)
        else:
            self.log.err(f"Wrong camera type. Pick rgb or semantic, not: {self.camera_type}")

        self.add_collision_sensor(self.a_vehicle)

        self.a_vehicle.apply_control(carla.VehicleControl(throttle=1.0, brake=1.0))

        time.sleep(0.5)

        while self.front_camera is None:
            time.sleep(0.01)

        self.a_vehicle.apply_control(carla.VehicleControl(brake=0.0))

        # Start the timer
        self.timer = time.time()

        # A frame from the spawn point
        return self.front_camera

    def step(self, action, vehicle_for_mc=None):
        """
        Method which creates an episode as a set of steps
        :param action: car's action
        :param vehicle_for_mc: carla.Vehicle class used only in human_performance_test.py
        :return:
        """
        self.clock.tick(fps)  # FPS

        if not self.manual_control:
            if self.action_space == 'continuous':
                self.car_control_continuous(action, self.a_vehicle)
            else:
                self.car_control_discrete(action, self.a_vehicle)
        else:
            self.a_vehicle = vehicle_for_mc

        if self.step_counter == 0:  # Only the first iteration
            self.thread = threading.Thread(target=self.chased_vehicle_movement, args=(), kwargs={}, daemon=True)
            self.thread.start()
            self.timer = time.time()

        self.step_counter += 1

        if sleep_time:
            # How many actions per sec?
            time.sleep(sleep_time)

        a_location, b_location = self.draw_movement(self.a_vehicle), self.b_vehicle.get_location()
        ab_distance = round(self.calculate_distance(a_location, b_location), 3)

        # angle = self.calculate_angle(self.a_vehicle, self.b_vehicle)

        timer = round(time.time() - self.timer, 3)  # How many seconds the episode has?

        # Done from a collision or a distnace
        reward, self.done = reward_function(self.collision_history_list, ab_distance=ab_distance, timer=timer)

        if not self.thread.is_alive():
            self.done = True  # The end of the episode

        return self.front_camera, reward, self.done

    def destroy_agents(self):
        """
        destroy each agent
        """
        for actor in self.actor_list:
            if hasattr(actor, 'is_listening') and actor.is_listening:
                actor.stop()
            if actor.is_alive:
                actor.destroy()
        self.actor_list.clear()
