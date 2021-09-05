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

Examples of some fundamental road manoeuvres:
<p align="left">
  <img src="Gif/chase/sc1/scenario1.gif" width="400"/>
  <img src="Gif/chase/sc3/scenario3.gif" width="400"/>
  <img src="Gif/chase/sc5/sc5.gif" width="400"/>
</p>

## Technologies
* Python 3.7
* PyTorch 1.8.1+cu111
* Carla 0.9.10 (Probably compatible with the latest versions)

