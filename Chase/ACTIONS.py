from dataclasses import dataclass


@dataclass()
class ACTIONS:
    forward = 0
    forward_left = 1
    forward_right = 2
    brake = 3
    brake_left = 4
    brake_right = 5

    ACTION_CONTROL = {
        # acc, br, steer
        0: [1, 0, 0],  # forward
        1: [1, 0, -0.5],  # forward left
        2: [1, 0, 0.5],  # forward right
        3: [0, 1, 0],  # brake
        4: [0, 1, -0.5],  # brake left
        5: [0, 1, 0.5],  # brake right
    }

    ACTIONS_NAMES = {
        0: 'forward',
        1: 'forward_left',
        2: 'forward_right',
        3: 'brake',
        4: 'brake_left',
        5: 'brake_right',
    }

    ACTIONS_VALUES = {y: x for x, y in ACTIONS_NAMES.items()}
