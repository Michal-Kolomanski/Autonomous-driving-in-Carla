The goal of this venture is to develop models capable of completing a variety of autonomous tasks within the Carla simulator using reinforcement learning methods. The repository is divided into two distinct projects which are still being developed.

# A to B
Autonomous ride from point A to B.
Trained models (see A_to_B/final_models) are able to correctly perform fundamental road manoeuvres such as driving straight, turning left and right.
Using this knowledge, the models can learn how to beat different scenarios in a reasonable amount of time, depending on the scenario's complexity.

Examples of some fundamental road manoeuvres:
<p align="left">
  <img src="Gif/a_b/sc1/scenario1.gif" width="400"/>
  <img src="Gif/a_b/sc3/scenario3.gif" width="400"/>
  <img src="Gif/a_b/sc4/scenario4.gif" width="400"/>
  <img src="Gif/a_b/sc5/scenario5.gif" width="400"/>
</p>

Example of a more challenging scenario:
<p align="centre">
  <img src="Gif/a_b/sc7/scenario7.gif" width="400"/>
</p>

# Chase
Autonomous chase of a fleeing vehicle.
This project followed a similar approach to learning. However, autonomous chasing proved to be more challenging than getting from point A to point B, and work on training a satisfactory model continues.

Examples of some fundamental chase manoeuvres:
<p align="left">
  <img src="Gif/chase/sc1/scenario1.gif" width="400"/>
  <img src="Gif/chase/sc3/scenario3.gif" width="400"/>
  <img src="Gif/chase/sc5/scenario5.gif" width="400"/>
</p>

## How to run?
1. Provide the paths to the Carla executable and egg files in the file *settings.py*,
2. Run *a2c_rgb.py* file.

## Requirements
* Python 3.7.x
* Pipenv
* Git
* Carla 0.9.10

The pipfile contains a list of all essential project packages. For information on how to install these, see the [setup](#setup) section.
If you wish to use a GPU with Pytorch, you'll need a device that supports CUDA.

## Setup
1. Install git, python and pipenv
2. Clone this repository and navigate to its root directory
```bash
git clone https://github.com/Michal-Kolomanski/Autonomous-driving-in-Carla
```
3. Install all required project packages by executing
```bash
pipenv install --dev
```

4. To open project virtual environment shell, type:
```bash
pipenv shell
```