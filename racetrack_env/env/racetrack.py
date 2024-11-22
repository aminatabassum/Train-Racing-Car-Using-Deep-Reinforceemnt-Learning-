import numpy as np
import math
from enum import Enum
import matplotlib.pyplot as plt
from collections import deque


class Config:
    STEERING_ANGLE = 0.5
    ALPHA = 0.5
    BASE_SPEED = 1.0
    MAX_SPEED = 5
    MAX_TIME_STEPS = 1000

    ON_ROAD_REWARD = 1
    ON_WIN_REWARD = 50
    ON_GRASS_REWARD = -3
    ON_SOIL_REWARD = -5
    SAME_ACTION_REWARD = -10


class PatchType(Enum):
    SOIL = 1
    GRASS = 2
    ROAD = 3
    SELF_CAR = 4


class RaceTrack:
    def __init__(self, track_file_path, starting_position, goal_positions, eye_sight, model_type):
        self.__track, self.__start_state, self.__goal_pos = self.__read_track(
            track_file_path, starting_position, goal_positions
        )
        self.__model_type = model_type

        self.__eye_sight = eye_sight
        self.__steering_angle = 0

        self.__t_gas = 0
        self.__t_brake = 0
        self.__curr_pos = None
        self.__curr_speed = [Config.BASE_SPEED, Config.BASE_SPEED]
        self.__curr_speed_total = 0
        self.__is_gas = False
        self.__is_brake = False
        self.__t = 0
        self.__curr_speeds = []
        self.__heading = [0, 1]
        self.action_space = {"noop": 0, "left": 1, "right": 2, "gas": 3, "brake": 4}
        self.action_names = ["Noop", "left", "right", "gas", "brake"]

        self.__actions_taken = deque(maxlen=5)

    def __check_valid_pos(self, track, pos):
        return 0 <= pos[0] < track.shape[0] and 0 <= pos[1] < track.shape[1]

    def __read_track(self, track_file_path, starting_position, goal_position):
        track = np.loadtxt(track_file_path, dtype=int)

        if (
            self.__check_valid_pos(track, starting_position)
            and track[starting_position] == PatchType.ROAD.value
        ):
            init_state = starting_position
        else:
            raise ValueError("Starting position is out of bounds!")

        for goal_pos in goal_position:
            if (
                self.__check_valid_pos(track, goal_pos)
                and track[goal_pos] == PatchType.ROAD.value
            ):
                continue
            else:
                raise ValueError("Goal position is out of bounds!")

        return track, init_state, goal_position

    def __to_shape(self, a, shape):
        y_, x_ = shape
        y, x = a.shape
        y_pad = y_ - y
        x_pad = x_ - x
        return np.pad(
            a,
            (
                (y_pad // 2, y_pad // 2 + y_pad % 2),
                (x_pad // 2, x_pad // 2 + x_pad % 2),
            ),
            mode="constant",
            constant_values=0,
        )

    def __get_state(self, pos):
        track_copy = self.__track.copy()
        track_copy[pos] = PatchType.SELF_CAR.value
        for goal_pos in self.__goal_pos:
            track_copy[goal_pos] = 10

        area_around = track_copy[
            pos[0] - self.__eye_sight : pos[0] + self.__eye_sight + 1,
            pos[1] - self.__eye_sight : pos[1] + self.__eye_sight + 1,
        ]

        if area_around.shape != (self.__eye_sight * 2 + 1, self.__eye_sight * 2 + 1):
            area_around = self.__to_shape(
                area_around, (self.__eye_sight * 2 + 1, self.__eye_sight * 2 + 1)
            )

        return area_around.flatten() if self.__model_type == "numerical" else np.expand_dims(area_around, axis=0)

    def reset(self):
        self.__curr_pos = self.__start_state
        self.__steering_angle = 0
        self.__curr_speed = [Config.BASE_SPEED, Config.BASE_SPEED]
        self.__actions_taken = deque(maxlen=5)
        self.__t_gas = 0
        self.__t_brake = 0
        self.__is_gas = False
        self.__is_brake = False
        self.__t = 0
        current_state = self.__get_state(self.__curr_pos)
        return (current_state, False)

    def render(self, mode="see"):
        track_copy = self.__track.copy()
        track_copy[self.__curr_pos] = PatchType.SELF_CAR.value

        for goal_pos in self.__goal_pos:
            track_copy[goal_pos] = 10

        if mode == "see":
            plt.imshow(track_copy)
            plt.show()
            return None
        elif mode == "rgb_array":
            return track_copy

    def step(self, action):
        if self.__curr_pos is None:
            raise ValueError("Reset the environment before using it!")

        # Push the most recent action into the deque
        self.__actions_taken.append(action)

        self.__curr_pos = list(self.__curr_pos)
        if action == 0:
            # NOOP
            self.__curr_pos[0] += int(max(self.__curr_speed[0], 0))
            self.__curr_pos[1] += int(max(self.__curr_speed[1], 0))
            self.__is_gas = False
            self.__is_brake = False
        elif action == 1:
            # LEFT
            self.__steering_angle = -Config.STEERING_ANGLE
            self.__is_gas = False
            self.__is_brake = False
        elif action == 2:
            # RIGHT
            self.__steering_angle = Config.STEERING_ANGLE + (
                math.sqrt(self.__heading[0] ** 2 + self.__heading[1] ** 2)
            )
            self.__is_gas = False
            self.__is_brake = False
        elif action == 3:
            # GAS
            if self.__t_gas != self.__t and not self.__is_gas:
                self.__t_gas = self.__t

            self.__curr_speed[0] += min(
                math.floor(abs(self.__t - self.__t_gas) * Config.ALPHA),
                Config.MAX_SPEED,
            )
            self.__curr_speed[1] += min(
                math.floor(abs(self.__t - self.__t_gas) * Config.ALPHA),
                Config.MAX_SPEED,
            )
            self.__curr_pos[0] += math.floor(
                int(max(self.__curr_speed[0], 0))
                * self.__heading[0]
                * math.sin(self.__steering_angle)
            )
            self.__curr_pos[1] += math.floor(
                int(max(self.__curr_speed[1], 0))
                * self.__heading[1]
                * math.cos(self.__steering_angle)
            )

            self.__is_gas = True
            self.__is_brake = False
            self.__t_brake = self.__t
        elif action == 4:
            # BRAKE
            if self.__t_brake != self.__t and not self.__is_brake:
                self.__t_brake = self.__t

            self.__curr_speed[0] -= math.floor(
                abs(self.__t - self.__t_brake)
                * Config.ALPHA
                * math.cos(self.__steering_angle)
            )
            self.__curr_speed[1] -= math.floor(
                abs(self.__t - self.__t_brake)
                * Config.ALPHA
                * math.sin(self.__steering_angle)
            )
            self.__curr_speed_total = math.sqrt(
                self.__curr_speed[0] ** 2 + self.__curr_speed[1] ** 2
            )
            self.__curr_pos[0] += int(
                max(self.__curr_speed[0], 0)
                * self.__heading[0]
                * math.sin(self.__steering_angle)
            )
            self.__curr_pos[1] += int(
                max(self.__curr_speed[1], 0)
                * self.__heading[1]
                * math.cos(self.__steering_angle)
            )

            self.__is_gas = False
            self.__is_brake = True
            self.__t_gas = self.__t

        # If agent tries to go outside the track, reset it
        self.__curr_pos = tuple(self.__curr_pos)
        if not self.__check_valid_pos(self.__track, self.__curr_pos):
            next_state, done = self.reset()
            reward = -50
            return next_state, reward, done, {}, {}

        self.__t += 1
        self.__curr_speeds.append(self.__curr_speed_total)

        next_state = self.__get_state(self.__curr_pos)
        done = self.__curr_pos in self.__goal_pos or self.__t >= Config.MAX_TIME_STEPS

        # Reward computation
        reward = None
        if self.__curr_pos in self.__goal_pos:
            reward = Config.ON_WIN_REWARD
        elif self.__track[self.__curr_pos] == PatchType.ROAD.value:
            reward = Config.ON_ROAD_REWARD
        elif self.__track[self.__curr_pos] == PatchType.GRASS.value:
            reward = Config.ON_GRASS_REWARD
        elif self.__track[self.__curr_pos] == PatchType.SOIL.value:
            reward = Config.ON_SOIL_REWARD

        # Check last 5 actions and if they are the same then set reward to -50
        if len(self.__actions_taken) == 5:
            actions = [
                1 if action in [1, 2] else action for action in self.__actions_taken
            ]
            if len(set(actions)) == 1:
                reward = Config.SAME_ACTION_REWARD

        return (next_state, reward, done, {}, {})
