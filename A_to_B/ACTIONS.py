from dataclasses import dataclass


@dataclass()
class ACTIONS:
    forward = 0
    # left = 1
    # right = 2
    forward_left = 1
    forward_right = 2
    brake = 3
    brake_left = 4
    brake_right = 5
    # no_action = 8

    # ACTION_CONTROL = {
    #     #  acc,  steer
    #     0: [1.0, 0.0],  # forward
    #     1: [0.0, -0.5],  # left
    #     2: [0.0, 0.5],  # right
    #     3: [1.0, -0.5],  # forward left
    #     4: [1.0, 0.5],  # forward right
    #     5: [-0.5, 0.0],  # brake
    #     6: [-0.5, -0.5],  # brake left
    #     7: [-0.5, 0.5],  # brake right
    #     8: [0.0, 0.0],  # no action
    # }

    # ACTION_CONTROL = {
    #     # acc, br, steer
    #     0: [1.0, 0.0, 0.0],  # forward
    #     1: [0.0, 0.0, -0.5],  # left
    #     2: [0.0, 0.0, 0.5],  # right
    #     3: [1.0, 0.0, -0.5],  # forward left
    #     4: [1.0, 0.0, 0.5],  # forward right
    #     5: [0.0, 0.5, 0.0],  # brake
    #     6: [0, 0.5, -0.5],  # brake left
    #     7: [0, 0.5, 0.5],  # brake right
    #     8: [0.0, 0.0, 0.0],  # no action
    # }

    ACTION_CONTROL = {
        # acc, br, steer
        0: [1, 0, 0],  # forward
        # 1: [0, 0, -1],  # left
        # 2: [0, 0, 1],  # right
        1: [1, 0, -0.5],  # forward left
        2: [1, 0, 0.5],  # forward right
        3: [0, 1, 0],  # brake
        4: [0, 1, -0.5],  # brake left
        5: [0, 1, 0.5],  # brake right
        # 8: [0, 0, 0],  # no action
    }

    ACTIONS_NAMES = {
        0: 'forward',
        # 1: 'left',
        # 2: 'right',
        1: 'forward_left',
        2: 'forward_right',
        3: 'brake',
        4: 'brake_left',
        5: 'brake_right',
        # 8: 'no_action',
    }

    ACTIONS_VALUES = {y: x for x, y in ACTIONS_NAMES.items()}
