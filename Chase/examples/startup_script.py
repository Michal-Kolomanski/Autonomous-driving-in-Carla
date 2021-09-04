import glob
import os
import sys
import time

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


import carla
import numpy as np
import subprocess
import psutil

class Env():
    def __init__(self,  carla_dir, quality, town_number, actors, car_model):
        self.spawn_point = 1
        self.build_carla(400, 400, carla_dir, quality, town_number)
        for i in range(int(actors)):
            print(i)
            self.transform = np.random.choice(self.spawn_points)
            self.spawn_actor(self.transform, car_model)

    def build_carla(self, resX, resY, carla_dir, quality, town_number):
        subprocess.Popen('CarlaUE4.exe' + f' -windowed -carla-server -benchmark -fps=20 -quality-level=' + quality,
                          cwd=carla_dir, shell=True)
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(60.0)
        self.actor_list = []
        self.client.load_world('Town' + town_number)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.resX = resX
        self.resY = resY
        map = self.world.get_map()
        self.spawn_points = map.get_spawn_points()

    def spawn_actor(self, spawn_point, car_model):
        while True:
            try:
                self.vehicle = self.world.try_spawn_actor(self.blueprint_library.filter(car_model)[0], spawn_point)
                self.actor_list.append(self.vehicle)
                print(self.vehicle.get_location())
                break
            except:
                time.sleep(0.01)




def main():
    carla_dir = input("Pass CarlaUE4.exe directory or './' if the script is in the folder with it. \n")
    quality = input("Pass desired quality of the map. Options are: 'Low', 'High', 'Epic'. \n")
    town_number = input("Pass the number of the map. Options are: '01', '02', '03'. \n")
    actors = input("Pass the number of actors to be spawned. \n")
    car_model = input("Pass the car model you want. '1' for Audi, '2' for BMW, '3' for Tesla. \n")
    
    Env(carla_dir, quality, town_number, actors, car_model)

if __name__ == '__main__':
   main()