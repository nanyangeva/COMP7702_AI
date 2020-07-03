from support.problem_spec import *
from support.robot_config import *
from support.angle import *
from tester import *
import example_search
import os
import numpy as np
import copy
import math
import datetime

os.getcwd()

class Solver:
    """The process is used to solve the problem and return the result."""

    def __init__(self, input_file):
        # generate the problem spec
        self.problem_spec = ProblemSpec(input_file)
        self.initial = self.problem_spec.initial  # RobotConfig obj
        self.goal = self.problem_spec.goal  # RobotConfig.obj
        self.obstacles = self.problem_spec.obstacles  # obstacles in Problem Spec
        self.num_segments = self.problem_spec.num_segments
        self.mini_lengths = self.problem_spec.min_lengths
        self.max_lengths = self.problem_spec.max_lengths
        self.config_list = [self.initial]
        self.ee1_grappled = self.initial.ee1_grappled
        self.ee2_grappled = self.initial.ee2_grappled

    def get_config_tuple(self, config: RobotConfig):
        config_params = []
        for i in range(len(config.points)):
            config_params.append(config.points[i][0])
            config_params.append(config.points[i][1])
        return tuple(config_params)

    def generate_single_random_RobotConfig(self, based_config):
        """generate out the new random config for a based_config.
        input: based_config
        output: a new random config has same ee1x,ee1y and ee1_grappled, ee2_grappled with the based_config."""
        ee1x, ee1y = based_config.get_ee1()
        ee2x, ee2y = based_config.get_ee2()
        # generate ee1_length
        config_lengths = []
        for i in range(self.num_segments):
            random_length = np.random.uniform(self.mini_lengths[i], self.max_lengths[i])
            # print(random_length)
            config_lengths.append(random_length)
        # generate ee1_angles
        config_angles = []
        random_degree = np.random.randint(-165, 165, self.num_segments)
        for i in range(self.num_segments):
            random_angle = Angle(degrees=random_degree[i])
            config_angles.append(random_angle)
        # generate config
        if based_config.ee1_grappled:
            random_config = make_robot_config_from_ee1(ee1x, ee1y, config_angles, config_lengths, ee1_grappled=True)
        else:
            random_config = make_robot_config_from_ee2(ee2x, ee2y, config_angles, config_lengths, ee2_grappled=True)
        return random_config

    def generate_random_RobotConfig(self, based_config):
        """random generate a valid state (RobotConfig) based on the based_config.
        checking rules : (1)angle_constraints (2)collide with obstacle (3)collide with itself (4) out of environment bounds"""
        random_config = self.generate_single_random_RobotConfig(based_config)
        while not(self.valid_config(random_config)):
            random_config = self.generate_single_random_RobotConfig(based_config)
        return random_config


    # def generate_bunch_RobotConfig(self, n):
    #     """random generate n configs
    #     return: config dictionary"""
    #     bunch_config = {}
    #     for i in range(n):
    #         config = self.generate_random_RobotConfig()
    #         bunch_config[config] = config
    #     return bunch_config

    def generate_single_RobotConfig_between_obstacle(self):
        """generate random robot config with end point near obstacle."""
        # new
        while True:
            random_config = self.generate_single_random_RobotConfig()
            ee1x, ee1y = self.initial.get_ee1()
            ee2x, ee2y = self.initial.get_ee2()
            # generate config_neighbour_length
            config_lengths_neighbour = random_config.lengths

            config_angles_neighbour = []
            for i in range(self.num_segments):
                random_radians = np.random.uniform(-0.05, 0.05)
                config_angles_neighbour.append(random_config.ee1_angles[i] + Angle(radians=random_radians))

            # generate config
            if self.ee1_grappled:
                config_neighbour = make_robot_config_from_ee1(ee1x, ee1y, config_angles_neighbour,
                                                              config_lengths_neighbour, ee1_grappled=True)
            else:
                config_neighbour = make_robot_config_from_ee2(ee2x, ee2y, config_angles_neighbour,
                                                              config_lengths_neighbour, ee2_grappled=True)

            if self.valid_config(random_config) and not (
            test_obstacle_collision(config_neighbour, self.problem_spec, self.obstacles)):
                return random_config
            elif not (
            test_obstacle_collision(config_neighbour, self.problem_spec, self.obstacles)) and self.valid_config(
                    config_neighbour):
                return config_neighbour

    def generate_bunch_config_in_between_obstacles(self, based_config, goal_config, n):
        """random generate n configs
        return: config dictionary"""
        # print('Points between obstacles')
        gaps = self.find_gap_in_between_obstacles()
        bunch_config = {}
        i = 0

        # if len(gaps) == 0:
        #     while i < n:
        #         config = self.generate_random_RobotConfig(based_config)
        #         point = self.get_config_non_grappled_ee_point(config)
        #         bunch_config[config] = config
        #         print(round(point[0],2), ",", round(point[1],2))
        #         i = i + 1
        #
        #     return bunch_config
        #
        # for gap in gaps:
        #     i = 0
        #     while i < n:
        #         config = self.generate_random_RobotConfig(based_config)
        #         point = self.get_config_non_grappled_ee_point(config)
        #         if point[0] >= gap[0] and point[0] <= gap[1] and point[1] >= gap[2] and point[1] <= gap[3]:
        #             bunch_config[config] = config
        #             print(round(point[0],2), ",", round(point[1],2))
        #             i = i + 1
        #
        # return bunch_config

        while i < n:
            if len(gaps) == 0:
                config = self.generate_random_RobotConfig(based_config)
                point = self.get_config_non_grappled_ee_point(config)
                bunch_config[config] = config
                # print(round(point[0], 2), ",", round(point[1], 2))
                i = i + 1
            else:
                for gap in gaps:
                    config = self.generate_random_RobotConfig(based_config)
                    point = self.get_config_non_grappled_ee_point(config)
                    if point[0] >= gap[0] and point[0] <= gap[1] and point[1] >= gap[2] and point[1] <= gap[3]:
                        bunch_config[config] = config
                        # print(round(point[0], 2), ",", round(point[1], 2))
                        i = i + 1
        return bunch_config

    def generate_bunch_config_outside_obstacles(self, based_config, goal_config, n, direction):
        """random generate n configs
        return: config dictionary"""

        # print('Points outside obstacles')

        grappled_ee_point = self.get_config_grappled_ee_point(based_config)
        non_grappled_ee_point = self.get_config_non_grappled_ee_point(goal_config)
        non_grappled_ee_arm_length = self.get_config_non_grappled_ee_arm_length(goal_config)
        gaps = self.find_gap_in_between_obstacles()
        bunch_config = {}
        i = 0

        # if len(gaps) == 0:
        #     while i < n:
        #         config = self.generate_random_RobotConfig(based_config)
        #         bunch_config[config] = config
        #         point = self.get_config_non_grappled_ee_point(config)
        #         print(round(point[0], 2), ",", round(point[1], 2))
        #     return bunch_config
        #
        # for gap in gaps:
        #     i = 0
        #     while i < n:
        #         config = self.generate_random_RobotConfig(based_config)
        #         point = self.get_config_non_grappled_ee_point(config)
        #         if direction == 'X':
        #             if point[0] <= gap[0] or point[0] >= gap[1]:
        #                 bunch_config[config] = config
        #                 print("Direction X:", round(point[0],2), ",", round(point[1],2))
        #                 i = i + 1
        #         if direction == 'Y':
        #             if point[1] <= gap[2] or point[1] >= gap[3]:
        #                 bunch_config[config] = config
        #                 print("Direction Y:",round(point[0],2), ",", round(point[1],2))
        #                 i = i + 1
        #
        # return bunch_config

        while i < n:
            if len(gaps) == 0:
                config = self.generate_random_RobotConfig(based_config)
                point = self.get_config_non_grappled_ee_point(config)
                bunch_config[config] = config
                # print(round(point[0], 2), ",", round(point[1], 2))
                i = i + 1
            else:
                for gap in gaps:
                    config = self.generate_random_RobotConfig(based_config)
                    point = self.get_config_non_grappled_ee_point(config)
                    if direction == 'X':
                        if point[0] <= gap[0] or point[0] >= gap[1]:
                            bunch_config[config] = config
                            # print("Direction X:", round(point[0], 2), ",", round(point[1], 2))
                            i = i + 1
                    if direction == 'Y':
                        if point[1] <= gap[2] or point[1] >= gap[3]:
                            bunch_config[config] = config
                            # print("Direction Y:", round(point[0], 2), ",", round(point[1], 2))
                            i = i + 1

        return bunch_config

    def find_gap_in_between_obstacles(self):
        """random generate n configs
        return: config dictionary"""
        bunch_config = {}
        i = 0
        j = 0
        gaps = set()
        if len(self.obstacles) < 2:
            return gaps
        for i in range(len(self.obstacles)):
            for j in range(len(self.obstacles)):
                if i >= j:
                    continue
                else:
                    if self.obstacles[i].x2 < self.obstacles[j].x1:
                        x_from = self.obstacles[i].x2
                        x_to = self.obstacles[j].x1
                    elif self.obstacles[i].x2 == self.obstacles[j].x1:
                        x_from = self.obstacles[i].x2
                        x_to = self.obstacles[j].x1
                    elif self.obstacles[i].x1 > self.obstacles[j].x2:
                        x_from = self.obstacles[j].x2
                        x_to = self.obstacles[i].x1
                    elif self.obstacles[i].x1 == self.obstacles[j].x2:
                        x_from = self.obstacles[j].x2
                        x_to = self.obstacles[i].x1
                    else:
                        x_from = max(self.obstacles[i].x1, self.obstacles[j].x1)
                        x_to = min(self.obstacles[i].x2, self.obstacles[j].x2)
                    if self.obstacles[i].y2 < self.obstacles[j].y1:
                        y_from = self.obstacles[i].y2
                        y_to = self.obstacles[j].y1
                    elif self.obstacles[i].y2 == self.obstacles[j].y1:
                        y_from = self.obstacles[i].y2
                        y_to = self.obstacles[j].y1
                    elif self.obstacles[i].y1 > self.obstacles[j].y2:
                        y_from = self.obstacles[j].y2
                        y_to = self.obstacles[i].y1
                    elif self.obstacles[i].y1 == self.obstacles[j].y2:
                        y_from = self.obstacles[i].y1
                        y_to = self.obstacles[j].y2
                    else:
                        y_from = max(self.obstacles[i].y1, self.obstacles[j].y1)
                        y_to = min(self.obstacles[i].y2, self.obstacles[j].y2)
            gaps.add((x_from, x_to, y_from, y_to))
        return gaps


    def prm(self, initial_config, goal_config, initial_graph, initial_distance_from_config,  n, config_distance, collision_free_distance):
        """generate a dictionary to generate out graph with configs
        input: bunch_configs  [config, config, config,...]
        output: {config:[config,config], ...}"""
        bunch_configs = {}
        bunch_configs_x = self.generate_bunch_config_outside_obstacles(initial_config, goal_config, n, 'X')
        bunch_configs_y = self.generate_bunch_config_outside_obstacles(initial_config, goal_config, n, 'Y')
        bunch_configs_near_obstacles = self.generate_bunch_config_in_between_obstacles(initial_config, goal_config, n)
        bunch_configs.update(bunch_configs_x)
        bunch_configs.update(bunch_configs_y)
        bunch_configs.update(bunch_configs_near_obstacles)

        bunch_configs_previous = {}
        for k, v in initial_graph.items():
            bunch_configs_previous[k] = k

        bunch_configs.update(bunch_configs_previous)

        # add the initial and goal configs
        initial_vertex_count = len(initial_graph)
        if initial_vertex_count == 0:
            bunch_configs[initial_config] = initial_config
            bunch_configs[goal_config] = goal_config

        # calculate distance between each pair of config
        distance_list = []
        distance_from_config = {}
        distance_from_config.update(initial_distance_from_config)
        for config in bunch_configs:
            if config in distance_from_config:
                distances_to_config = distance_from_config[config]
            else:
                distances_to_config = {}
            for config_another in bunch_configs:
                if test_config_equality(config, config_another, self.problem_spec):
                    distances_to_config[config_another] = 0
                else:
                    distance = self.calculate_config_distance(config, config_another)
                    distances_to_config[config_another] = distance
                    distance_list.append(distance)
            distance_from_config[config] = distances_to_config


        # # generate graph
        # graph = {}
        # graph.update(initial_graph)
        # for config in bunch_configs:
        #     if config not in graph:
        #         graph[config] = []
        #     distances_to_config = {}
        #     distances_to_config = distance_from_config[config]
        #     for config_another, v in distances_to_config.items():
        #         if 0 < v <= config_distance:
        #             if config_another not in graph[config]:
        #                 graph[config].append(config_another)
        #                 # if config_another not in graph:
        #                 #     graph[config_another] = []
        #                 # if config not in graph[config_another]:
        #                 #     graph[config_another].append(config)
        #         if v == -1:
        #             q_new_config = self.generate_q_new_config(config, config_another, 0.3)
        #             # if self.line_collision_free(config, q_new_config):
        #             if self.collision_free(config, q_new_config):
        #                 if self.calculate_config_distance(config, q_new_config) <= config_distance and q_new_config not in graph[config]:
        #                     graph[config].append(q_new_config)
        #                 # if q_new_config not in graph:
        #                 #     graph[q_new_config] = []
        #                 # if config not in graph[q_new_config]:
        #                 #     graph[q_new_config].append(config)

        # generate graph

        distance_list.sort(reverse = False)
        threshold_distance = distance_list[int(0.1*len(distance_list))]


        graph = {}
        graph.update(initial_graph)
        for config in bunch_configs:
            if config not in graph:
                graph[config] = []
            for config_another, v in bunch_configs.items():
                distance = self.calculate_config_distance(config, config_another)
                if distance == 0 or test_config_equality(config, config_another, self.problem_spec):
                    continue

                if 0 < distance <= threshold_distance:
                    if self.collision_free(config, config_another):
                        graph[config].append(config_another)
                    else:
                        q_new_config = self.generate_q_new_config(config, config_another, 0.3)
                        distance = self.calculate_config_distance(config, q_new_config)
                        if distance <= config_distance and self.collision_free(config, q_new_config):
                            graph[config].append(q_new_config)

        return graph

    def rrt(self, initial_config, goal_config, initial_graph, n, delta_q):
        initial_vertex_count = len(initial_graph)
        graph = {}
        graph.update(initial_graph)

        if initial_vertex_count == 0:
            graph[initial_config] = []

        vertex_count = 0
        while vertex_count < n:
            # q_config is the randomly generated q node
            # p_config is each every graph nodes
            q_config = self.generate_random_RobotConfig(initial_config)
            p_config, v = self.find_nearest_graph_vertex(q_config, graph)
            q_new_config = self.generate_q_new_config(p_config, q_config, delta_q)
            if not (self.valid_config(q_new_config)):
                continue

            # if q_new_config exists already in the graph
            already_exists = False
            for k, v in graph.items():
                if test_config_equality(q_new_config, k, self.problem_spec):
                    already_exists = True
                    break
            if already_exists:
                continue
            # if q_new_config to p_config is a collision free path
            # if self.line_collision_free(p_config, q_new_config):
            if self.collision_free(p_config, q_new_config):
                # print("q new config is", self.get_config_non_grappled_ee_point(q_new_config))
                if p_config not in graph:
                    graph[p_config] = []
                # if the q_new_config is actually the goal config and initial config has a neighbour
                if test_config_equality(q_new_config, goal_config, self.problem_spec):
                # if self.calculate_config_distance(q_new_config, goal_config) < delta_q:
                    graph[p_config].append(goal_config)
                    vertex_count = vertex_count + 1
                    return graph, len(graph), True
                else:
                    graph[p_config].append(q_new_config)
                    if q_new_config not in graph:
                        graph[q_new_config] = []
                    # graph[q_new_config].append(p_config)
                    vertex_count = vertex_count + 1
                # # if the q_new_config is actually the initial config and goal config has a neighbour
                # if test_config_equality(q_new_config, initial_config, self.problem_spec) and len(
                #         graph[goal_config]) > 0:
                #     print("RRT found initial config, vertex count = ", len(graph))
                #     return graph, len(graph), True
        print("RRT not solved, vertex count = ", len(graph))
        return graph, len(graph), False

    def generate_q_new_config(self, p: RobotConfig, q: RobotConfig, delta_q):
        p_vector = self.generate_config_vector(p)
        q_vector = self.generate_config_vector(q)
        q_new_vector = []
        for i in range(len(p_vector)):
            if isinstance(p_vector[i], Angle):
                radians_val = (q_vector[i].in_radians() - p_vector[i].in_radians())*delta_q + p_vector[i].in_radians()
                angle = Angle(radians= radians_val)
                q_new_vector.append(angle)
            else:
                q_new_vector.append((q_vector[i] - p_vector[i])*delta_q + p_vector[i])

        config_angles = []
        for i in range(len(p.ee1_angles)):
            config_angles.append(q_new_vector[i])

        config_lengths = []
        for i in range(len(p.ee1_angles), len(p.ee1_angles) + len(p.lengths)):
            config_lengths.append(q_new_vector[i])

        if p.ee1_grappled:
            random_config = make_robot_config_from_ee1(p.points[0][0], p.points[0][1], config_angles, config_lengths, ee1_grappled=True)
        else:
            random_config = make_robot_config_from_ee2(p.points[-1][0], p.points[-1][1], config_angles, config_lengths, ee2_grappled=True)

        return random_config


    def find_nearest_graph_vertex(self, config: RobotConfig, graph):
        distance_dict = {}
        for k, v in graph.items():
            # distance_dict[k] = self.calculate_config_non_grappled_ee_distance(config, k)
            distance_dict[k] = self.calculate_config_distance(config, k)
        min_key = min(distance_dict, key=lambda k: distance_dict[k])
        min_value = distance_dict[min_key]
        return min_key, min_value

    @staticmethod
    def calculate_point_distance(point1, point2):
        return np.sqrt(np.square(point1[0] - point2[0]) + np.square(point1[1] - point2[1]))

    def calculate_config_non_grappled_ee_distance(self, config1: RobotConfig, config2: RobotConfig):
        points_count = len(config1.points)
        if config1.ee1_grappled is True and config2.ee1_grappled is True:
            point1 = config1.points[points_count - 1]
            point2 = config2.points[points_count - 1]
            return self.calculate_point_distance(point1, point2)

        if config1.ee2_grappled is True and config2.ee2_grappled is True:
            point1 = config1.points[0]
            point2 = config2.points[0]
            return self.calculate_point_distance(point1, point2)

        return 9999
    def different_ee_goal_generator(self, based_config, point):
        """generate a config with the two ee_grapples as based_config.points[0] , point."""
        #  test:  initial_config = self.generate_random_RobotConfig()
        #         point = (0.6516529983330152, 0.25761455641847525)
        # step1: generate goal config
        is_valid_sample = False
        while not is_valid_sample:
            sample = self.generate_random_RobotConfig(based_config)
            # the config's ee_point
            ee_point = self.get_config_non_grappled_ee_point(sample)
            # the config's ee_point connected arm
            ee_arm_length = self.get_config_non_grappled_ee_arm_length(sample)
            if ee_point[0] >= point[0] - ee_arm_length and ee_point[0] <= point[0] + ee_arm_length and ee_point[1] >= point[1] - ee_arm_length and ee_point[1] <= point[1] + ee_arm_length:
                is_valid_sample = True

        # print("length:",sample.lengths)
        point0 = sample.points[-4]
        # print("point0:",point0)
        point1 = sample.points[-3]         # the inverse 3rd point
        # print("point1:",point1)
        # point2 = self.goal.points[-1]    #  the inverse 1st point
        point2 = point
        # print("point2:",point2)
        r1 = sample.lengths[-2]          #  the inverse 3rd length
        r2 = sample.lengths[-1]          #  the inverse 1st length
        print("r1:",r1)
        print("r2:",r2)

        if r1 + r2 > self.points_distance(point1,point2):
            print("ok")
            c1, c2 = self.circle_insec(point1,r1,point2,r2)
            # generate config for c1
            v1 = np.array(point1) - np.array(point0)
            v2 = np.array(c1) - np.array(point1)
            v3 = np.array(point2) - np.array(c1)

            angle1_c1 = math.acos(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))
            angle2_c1 = math.acos(np.dot(v2,v3)/(np.linalg.norm(v2)*np.linalg.norm(v3)))

            if based_config.ee1_grappled:
                angles = sample.ee1_angles[:-2]
                angles1 = -Angle(radians=angle1_c1)
                angles2 = -Angle(radians=angle2_c1)
                angles.extend([angles1,angles2])
                goal_config_1 = make_robot_config_from_ee1(based_config.get_ee1()[0], based_config.get_ee1()[1], angles, sample.lengths, ee1_grappled=True)
            else:
                angles = sample.ee2_angles[:-2]
                angles1 = -Angle(radians=angle1_c1)
                angles2 = -Angle(radians=angle2_c1)
                angles.extend([angles1,angles2])
                goal_config_1 = make_robot_config_from_ee2(based_config.get_ee1()[0], based_config.get_ee1()[1], angles, sample.lengths, ee2_grappled=True)

            # generate config for c2
            v1 = np.array(point1) - np.array(point0)
            v2 = np.array(c2) - np.array(point1)
            v3 = np.array(point2) - np.array(c2)

            angle1_c2 = math.acos(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))
            angle2_c2 = math.acos(np.dot(v2,v3)/(np.linalg.norm(v2)*np.linalg.norm(v3)))
            if based_config.ee1_grappled:
                angles = sample.ee1_angles[:-2]
                angles1 = -Angle(radians=angle1_c2)
                angles2 = -Angle(radians=angle2_c2)
                angles.extend([angles1,angles2])
                goal_config_2 = make_robot_config_from_ee1(based_config.get_ee1()[0], based_config.get_ee1()[1], angles, sample.lengths, ee1_grappled=True)
            else:
                angles = sample.ee2_angles[:-2]
                angles1 = -Angle(radians=angle1_c2)
                angles2 = -Angle(radians=angle2_c2)
                angles.extend([angles1,angles2])
                goal_config_2 = make_robot_config_from_ee2(based_config.get_ee1()[0], based_config.get_ee1()[1], angles, sample.lengths, ee2_grappled=True)

            if self.valid_config(goal_config_1):
                # print("goal_config_1_points:", goal_config_1.points)
                # print("c1:",c1)
                return goal_config_1
            if self.valid_config(goal_config_2):
                # print("goal_config_2_points:", goal_config_2.points)
                # print("c2:",c2)
                return goal_config_2
            else:
                return False
        else:
            return False

    def different_ee_goal_generator_success(self, based_config, point):
        # step2: success generate out a goal config for change ee1
        # new
        generator = False
        while generator == False:
            generator = self.different_ee_goal_generator(based_config, point)
        return generator

    def calculate_config_distance(self, config1: RobotConfig, config2: RobotConfig):
        vector1 = self.generate_config_vector(config1)
        vector2 = self.generate_config_vector(config2)
        distance = 0
        for i in range(len(vector1)):
            if isinstance(vector1[i], Angle):
                angle1 = vector1[i]
                angle1_radians = angle1.in_radians()
                angle2 = vector2[i]
                angle2_radians = angle2.in_radians()
                distance = distance + (angle1_radians - angle2_radians)*(angle1_radians - angle2_radians)
            else:
                distance = distance + (vector1[i] - vector2[i]) * (vector1[i] - vector2[i])
        distance = math.sqrt(distance)
        return distance

    @staticmethod
    def calculate_point_to_edge_distance(point, edge):
        p = edge[0]
        q = edge[1]
        a = q[1] - p[1]
        b = p[0] - q[0]
        c = q[0] * p[1] - p[0] * q[1]
        return abs(a * point[0] + b * point[1] + c) / math.sqrt(a * a + b * b)


    def get_config_non_grappled_ee_point(self, config: RobotConfig):
        points_count = len(config.points)
        if config.ee1_grappled is True:
            return config.points[points_count - 1]

        if config.ee2_grappled is True:
            return config.points[0]

    def get_config_grappled_ee_point(self, config: RobotConfig):
        points_count = len(config.points)
        if config.ee1_grappled is True:
            return config.points[0]

        if config.ee2_grappled is True:
            return config.points[points_count - 1]

    def get_config_non_grappled_ee_arm_length(self, config: RobotConfig):
        points_count = len(config.points)
        arm_count = points_count - 1
        if config.ee1_grappled is True:
            return config.lengths[arm_count - 1]

        if config.ee2_grappled is True:
            return config.lengths[0]

    def get_non_grappled_ee_point(self, config: RobotConfig):
        points_count = len(config.points)
        if config.ee1_grappled is True:
            point = config.points[points_count - 1]
            return point

        if config.ee2_grappled is True:
            point = config.points[0]
            return point

    def generate_config_vector(self, config):
        """generate config vector for the given config.
        config: config object
        return : [Angle,...,length,...]"""
        config_vector = []
        if self.ee1_grappled:
            config_vector.extend(config.ee1_angles)
            config_vector.extend(config.lengths)
        else:
            config_vector.extend(config.ee2_angles)
            config_vector.extend(config.lengths)
        return config_vector

    def valid_config(self, config):
        """check whether a config is valid or not.
        valid config : return(True)
        invalid config: return(False)
        # check whether the random meet requirement
        # (1) test_angle_constraints
        # (2) not collide with obstacle
        # (3) not collide with itself
        # (4) not out of the environment bounds : """
        return (test_angle_constraints(config, self.problem_spec)
                and test_obstacle_collision(config, self.problem_spec, self.obstacles)
                and test_self_collision(config, self.problem_spec)
                and test_environment_bounds(config))

    def collision_free(self, config1, config2):
        """ one step move forward from config_1.
        config_1 = RobotConfig([2, 2, 2], ee1x=3, ee1y=5,ee1_angles=[Angle(degrees=45), Angle(degrees=30), Angle(degrees=40)],ee1_grappled=True)
        config_2 = RobotConfig([2, 2, 2], ee1x=3, ee1y=5,ee1_angles=[Angle(degrees=30), Angle(degrees=30), Angle(degrees=-60)],ee1_grappled=True)
        note: the change of one step will not change config_1 and config_2 themselves"""
        # new

        # generate different steps
        config_new = copy.deepcopy(config1)
        diff = np.array(self.generate_config_vector(config_new)) - np.array(
            self.generate_config_vector(config2))  # [angle,angle,..l,l,..]

        angles_diff = diff[0:self.num_segments]
        angle_preminum = Angle(radians=0.001)
        steps = [item.in_radians() / angle_preminum.in_radians() for item in angles_diff]  # steps : list

        length_preminum = 0.001
        length_steps = diff[self.num_segments:] / length_preminum
        steps.extend(length_steps)  # steps: [0.0, -523.5987755983568, 0.0, 0.0, -499.99999999999994, 0.0]

        max_index = list(np.abs(steps)).index(max(np.abs(steps)))  # max_index : 1

        while not (test_config_equality(config_new, config2, self.problem_spec)):
            if config_new.ee1_grappled:
                # change the angels
                if max_index < self.num_segments:
                    if np.abs(steps[max_index]) > 1:
                        config_new.ee1_angles[max_index] += np.sign(steps[max_index]) * (-1) * angle_preminum
                        # update the steps
                        steps[max_index] += np.sign(steps[max_index]) * (-1)
                    else:
                        config_new.ee1_angles[max_index] = config2.ee1_angles[max_index]
                        # update the steps
                        steps[max_index] = 0
                # change the lengths
                else:
                    if np.abs(steps[max_index]) > 1:
                        config_new.lengths[max_index - self.num_segments] += np.sign(steps[max_index]) * (
                            -1) * length_preminum
                        # update the steps
                        steps[max_index] += np.sign(steps[max_index]) * (-1)
                    else:
                        config_new.lengths[max_index - self.num_segments] = config2.lengths[
                            max_index - self.num_segments]
                        # update the steps
                        steps[max_index] = 0
                # update the points
                config_new = RobotConfig(config_new.lengths, ee1x=config_new.get_ee1()[0], ee1y=config_new.get_ee1()[1],
                                         ee1_angles=config_new.ee1_angles,
                                         ee1_grappled=True)
                # print("0:", np.array(config_new.ee1_angles[0].in_degrees()))
                # print("1:", np.array(config_new.ee1_angles[1].in_degrees()))
                # print("2:", np.array(config_new.lengths[0]))
                # print("points:",config_new.points)

            else:
                # change the angels
                if max_index < self.num_segments:
                    if np.abs(steps[max_index]) > 1:
                        config_new.ee2_angles[max_index] += np.sign(steps[max_index]) * (-1) * angle_preminum
                        # update the steps
                        steps[max_index] += np.sign(steps[max_index]) * (-1)
                    else:
                        config_new.ee2_angles[max_index] = config2.ee2_angles[max_index]
                        # update the steps
                        steps[max_index] = 0
                # change the lengths
                else:
                    if np.abs(steps[max_index]) > 1:
                        config_new.lengths[max_index - self.num_segments] += np.sign(steps[max_index]) * (
                            -1) * length_preminum
                        # update the steps
                        steps[max_index] += np.sign(steps[max_index]) * (-1)
                    else:
                        config_new.lengths[max_index - self.num_segments] = config2.lengths[
                            max_index - self.num_segments]
                        # update the steps
                        steps[max_index] = 0

                # update the points
                config_new = RobotConfig(config_new.lengths, ee2x=config_new.get_ee1()[0], ee2y=config_new.get_ee1()[1],
                                         ee2_angles=config_new.ee1_angles,
                                         ee2_grappled=True)

            # update the max_index
            max_index = list(np.abs(steps)).index(max(np.abs(steps)))

            # check collision free
            if not (self.valid_config(config_new)):
                return False
        return True

    def line_collision_free(self, config_1, config_2):
        point1 = self.get_config_non_grappled_ee_point(config_1)
        point2 = self.get_config_non_grappled_ee_point(config_2)

        collision_found = False
        for i in range(len(self.obstacles)):
            for j in range(len(self.obstacles[i].edges)):
                if test_line_collision((point1, point2), self.obstacles[i].edges[j]):
                    collision_found = True
        return not collision_found

    def line_collision_free_by_points(self, point1, point2):

        collision_found = False
        for i in range(len(self.obstacles)):
            for j in range(len(self.obstacles[i].edges)):
                if test_line_collision((point1, point2), self.obstacles[i].edges[j]):
                    collision_found = True
        return not collision_found

    def graph(self, initial_config, goal_config, bunch_configs):
        # new
        """generate a dictionary to generate out graph with configs
        input: bunch_configs  [config, config, config,...]
        output: {config:[config,config], ...}"""

        # add the initial and goal configs
        bunch_configs.append(initial_config)
        bunch_configs.append(goal_config)

        # generate graph
        graph = {}
        # print("test_graph")
        # print(len(bunch_configs))
        for i in range(len(bunch_configs) - 1):
            print(i)
            for j in range(i + 1, len(bunch_configs)):
                if self.check_distance_of_two_configs_method1(bunch_configs[i], bunch_configs[j]) < 0.3:
                    if not (test_config_equality(bunch_configs[i], bunch_configs[j], self.problem_spec)) and \
                            self.collision_free(bunch_configs[i], bunch_configs[j]):
                        if bunch_configs[i] in graph.keys():
                            graph[bunch_configs[i]].append(bunch_configs[j])
                        else:
                            graph[bunch_configs[i]] = [bunch_configs[j]]
                        if bunch_configs[j] in graph.keys():
                            graph[bunch_configs[j]].append(bunch_configs[i])
                        else:
                            graph[bunch_configs[j]] = [bunch_configs[i]]
        return graph

    def BFS(self, graph):
        """apply BFS on the given bunch_configs
        input: graph dictionary"""
        # keep records of all visited nodes : [config1, config2]
        explored = []
        # keep records of need check nodes :  [[config,config],[config,config]]
        initial = self.initial
        goal_config = self.goal
        queue = [[initial]]

        while len(queue) > 0:
            # pop first node in queue                         # queue = [[config1]]
            node = queue.pop(0)  # node : [config1]
            if node[-1] not in explored:  # node[-1] : config1
                # add node to explored list
                explored.append(node[-1])
                # if the current node has children
                if node[-1] in graph.keys():
                    for child in graph[node[-1]]:  # for each child in children of a frontier
                        node_duplicate = copy.deepcopy(node)
                        node_duplicate.append(child)
                        if test_config_equality(child, goal_config, self.problem_spec):
                            return node_duplicate, True
                        else:
                            queue.append(node_duplicate)
                # if the current node has no child : do nothing
        return [], False

    # def bunch_BFS(self):
    #     bunch_configs_list = self.generate_bunch_RobotConfig(50)
    #     result = self.BFS(bunch_configs_list)
    #     while not (result):
    #         bunch_configs = self.generate_bunch_RobotConfig(50)
    #         bunch_configs_list.extend(bunch_configs)
    #         result = self.BFS(bunch_configs_list)
    #     return result

    def prm_bfs(self, graph):
        """apply BFS on the given bunch_configs
        input: graph dictionary"""
        # keep records of all visited nodes : [config1, config2]
        explored = []
        # keep records of need check nodes :  [[config,config],[config,config]]
        initial = self.initial
        goal_config = self.goal
        queue = [[initial]]

        while len(queue) > 0:
            # pop first node in queue                         # queue = [[config1]]
            node = queue.pop(0)  # node : [config1]
            if node[-1] not in explored:  # node[-1] : config1
                # add node to explored list
                explored.append(node[-1])
                # if the current node has children
                if node[-1] in graph.keys():
                    for child in graph[node[-1]]:  # for each child in children of a frontier
                        node_duplicate = copy.deepcopy(node)
                        node_duplicate.append(child)
                        if test_config_equality(child, goal_config, self.problem_spec):
                            return node_duplicate, True
                        else:
                            queue.append(node_duplicate)
                # if the current node has no child : do nothing
        return [], False

    def check_distance_of_two_configs_method1(self, config1, config2):
        """This method used for A* methodology.
        check the distance of two given configs.
        return: distance of the last point."""
        point1 = config1.points[-2]
        point2 = config2.points[-2]
        distance = abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])
        # print("point1:",point1)
        # print("point2:",point2)
        return distance

    def check_distance_of_two_configs_method2(self, config1, config2):
        """This method used for A* methodology.
        check the difference of two given configs by the config vector discrepancy.
        return: distance of the config_vectors."""
        vector1 = self.generate_config_vector(config1)
        vector2 = self.generate_config_vector(config2)
        diff = np.array(vector2) - np.array(vector1)
        distance = 0
        for i in range(self.num_segments):
            distance += abs(diff[i].in_radians())
        for i in range(self.num_segments, 2 * self.num_segments):
            distance += abs(diff[i])
        return distance

    def a_star(self, graph, initial_config, goal_config):
        """set up an a_star methodology to find the way."""
        # g(x) + h(x)
        # g(x) : the sum of vector_diff between two configs   # check_distance_of_two_configs_method2(self,config1,config2)
        # h(x) : the distance between the second inverse point of two configs  # check_distance_of_two_configs_method1(self,config1,config2)
        current_node = initial_config
        explored_nodes = [current_node]
        print(graph[current_node])
        # while len(graph[current_node])>0:
        while current_node in graph.keys():
            shortest_node = graph[current_node][0]
            shortest_f = self.check_distance_of_two_configs_method2(current_node, shortest_node) + \
                         self.check_distance_of_two_configs_method1(shortest_node, goal_config)
            for node in graph[current_node]:
                g = self.check_distance_of_two_configs_method2(current_node, node)
                h = self.check_distance_of_two_configs_method1(node, goal_config)
                f = g + h
                if f < shortest_f:
                    shortest_node = node
                    shortest_f = f
            explored_nodes.append(shortest_node)
            current_node = shortest_node
            if test_config_equality(current_node, goal_config, self.problem_spec):
                return explored_nodes
        return False

    def circle_insec(self, p1, r1, p2, r2):
        """input 2 circles and generate their intersection points
        return: intersection points"""
        x = p1[0]
        y = p1[1]
        R = r1
        a = p2[0]
        b = p2[1]
        S = r2
        d = math.sqrt((abs(a - x)) ** 2 + (abs(b - y)) ** 2)
        if d > (R + S) or d < (abs(R - S)):
            print("Two circles have no intersection")
            return
        elif d == 0 and R == S:
            print("Two circles have same center!")
            return
        else:
            A = (R ** 2 - S ** 2 + d ** 2) / (2 * d)
            h = math.sqrt(R ** 2 - A ** 2)
            x2 = x + A * (a - x) / d
            y2 = y + A * (b - y) / d
            x3 = round(x2 - h * (b - y) / d, 2)
            y3 = round(y2 + h * (a - x) / d, 2)
            x4 = round(x2 + h * (b - y) / d, 2)
            y4 = round(y2 - h * (a - x) / d, 2)
            print(x3, y3)
            print(x4, y4)
            c1 = np.array([x3, y3])
            c2 = np.array([x4, y4])
            return c1, c2

    def points_distance(self, point1, point2):
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def different_ee1_goal_generator(self,point):
        # step1: generate goal config
        sample = self.generate_random_RobotConfig()
        print("length:", sample.lengths)
        point0 = sample.points[-4]
        print("point0:", point0)
        point1 = sample.points[-3]  # the inverse 3rd point
        print("point1:", point1)
        point2 = point # the inverse 1st point
        print("point2:", point2)
        r1 = sample.lengths[-2]  # the inverse 3rd length
        r2 = sample.lengths[-1]  # the inverse 1st length
        print("r1:", r1)
        print("r2:", r2)

        if r1 + r2 > self.points_distance(point1, point2):
            print("ok")
            c1, c2 = self.circle_insec(point1, r1, point2, r2)
            # generate config for c1
            v1 = np.array(point1) - np.array(point0)
            v2 = np.array(c1) - np.array(point1)
            v3 = np.array(point2) - np.array(c1)
            # angle1_c1 = np.arccos(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))
            # angle2_c1 = np.arccos(np.dot(v2,v3)/(np.linalg.norm(v2)*np.linalg.norm(v3)))
            angle1_c1 = math.acos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            angle2_c1 = math.acos(np.dot(v2, v3) / (np.linalg.norm(v2) * np.linalg.norm(v3)))

            if self.initial.ee1_grappled:
                angles = sample.ee1_angles[:-2]
                angles1 = -Angle(radians=angle1_c1)
                angles2 = -Angle(radians=angle2_c1)
                angles.extend([angles1, angles2])
                goal_config_1 = make_robot_config_from_ee1(self.initial.get_ee1()[0], self.initial.get_ee1()[1], angles,
                                                           sample.lengths, ee1_grappled=True)
            else:
                angles = sample.ee2_angles[:-2]
                angles1 = -Angle(radians=angle1_c1)
                angles2 = -Angle(radians=angle2_c1)
                angles.extend([angles1, angles2])
                goal_config_1 = make_robot_config_from_ee2(self.initial.get_ee1()[0], self.initial.get_ee1()[1], angles,
                                                           sample.lengths, ee2_grappled=True)

            # generate config for c2
            v1 = np.array(point1) - np.array(point0)
            v2 = np.array(c2) - np.array(point1)
            v3 = np.array(point2) - np.array(c2)
            angle1_c2 = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            angle2_c2 = np.arccos(np.dot(v2, v3) / (np.linalg.norm(v2) * np.linalg.norm(v3)))
            if self.initial.ee1_grappled:
                angles = sample.ee1_angles[:-2]
                angles1 = -Angle(radians=angle1_c2)
                angles2 = -Angle(radians=angle2_c2)
                angles.extend([angles1, angles2])
                goal_config_2 = make_robot_config_from_ee1(self.initial.get_ee1()[0], self.initial.get_ee1()[1], angles,
                                                           sample.lengths, ee1_grappled=True)
            else:
                angles = sample.ee2_angles[:-2]
                angles1 = -Angle(radians=angle1_c2)
                angles2 = -Angle(radians=angle2_c2)
                angles.extend([angles1, angles2])
                goal_config_2 = make_robot_config_from_ee2(self.initial.get_ee1()[0], self.initial.get_ee1()[1], angles,
                                                           sample.lengths, ee2_grappled=True)

            if self.valid_config(goal_config_1):
                print("goal_config_1_points:", goal_config_1.points)
                print("c1:", c1)
                return goal_config_1
            if self.valid_config(goal_config_2):
                print("goal_config_2_points:", goal_config_2.points)
                print("c2:", c2)
                return goal_config_2
            else:
                return False
        else:
            return False

    def different_ee1_goal_generator_success(self,point):
        # step2: success generate out a goal config for change ee1
        generator = False
        while generator == False:
            generator = self.different_ee1_goal_generator(point)
        return generator

    def change_config_direction(self, config):
        # new
        # change the direction of config from ee1 to ee2
        """change the config_ee1 to config_ee2
        return : config_ee2"""

        # ee1 -> ee2
        if config.ee1_grappled:
            x, y = config.points[0]
            angles = config.ee1_angles[:]
            lengths = config.lengths[:]
            lengths.reverse()
            c_flipped = make_robot_config_from_ee2(x, y, angles, lengths)
        # ee2 -> ee1
        else:
            x, y = config.points[0]
            angles = config.ee2_angles[:]
            lengths = config.lengths[:]
            lengths.reverse()
            c_flipped = make_robot_config_from_ee1(x, y, angles, lengths)
        return c_flipped

    def collision_free_steps_generator(self, config1, config2):
        # new
        """generate the moving steps between two collission-free configs.
        input: two configs (need collision-free)
        return: return a list of configs with slight move.
        note: result list will not include the config2. """

        result_list = []
        config_new = copy.deepcopy(config1)
        diff = np.array(self.generate_config_vector(config_new)) - np.array(
            self.generate_config_vector(config2))  # [angle,angle,..l,l,..]
        angles_diff = diff[0:self.num_segments]
        angle_preminum = Angle(radians=0.001)
        steps = [item.in_radians() / angle_preminum.in_radians() for item in angles_diff]  # steps : list
        length_preminum = 0.001
        length_steps = diff[self.num_segments:] / length_preminum
        steps.extend(length_steps)  # steps: [0.0, -523.5987755983568, 0.0, 0.0, -499.99999999999994, 0.0]
        max_index = list(np.abs(steps)).index(max(np.abs(steps)))  # max_index : 1

        # add the first element
        result_list.append(config_new)
        while not (test_config_equality(config_new, config2, self.problem_spec)):
            if config_new.ee1_grappled:
                # change the angels
                if max_index < self.num_segments:
                    if np.abs(steps[max_index]) > 1:
                        config_new.ee1_angles[max_index] += np.sign(steps[max_index]) * (-1) * angle_preminum
                        # update the steps
                        steps[max_index] += np.sign(steps[max_index]) * (-1)
                    else:
                        config_new.ee1_angles[max_index] = config2.ee1_angles[max_index]
                        # update the steps
                        steps[max_index] = 0
                # change the lengths
                else:
                    if np.abs(steps[max_index]) > 1:
                        config_new.lengths[max_index - self.num_segments] += np.sign(steps[max_index]) * (
                            -1) * length_preminum
                        # update the steps
                        steps[max_index] += np.sign(steps[max_index]) * (-1)
                    else:
                        config_new.lengths[max_index - self.num_segments] = config2.lengths[
                            max_index - self.num_segments]
                        # update the steps
                        steps[max_index] = 0
                # update the points
                config_new = RobotConfig(config_new.lengths, ee1x=config_new.get_ee1()[0], ee1y=config_new.get_ee1()[1],
                                         ee1_angles=config_new.ee1_angles,
                                         ee1_grappled=True)
                # print("0:", np.array(config_new.ee1_angles[0].in_degrees()))
                # print("1:", np.array(config_new.ee1_angles[1].in_degrees()))
                # print("2:", np.array(config_new.lengths[0]))
                # print("points:",config_new.points)

            else:
                # change the angels
                if max_index < self.num_segments:
                    if np.abs(steps[max_index]) > 1:
                        config_new.ee2_angles[max_index] += np.sign(steps[max_index]) * (-1) * angle_preminum
                        # update the steps
                        steps[max_index] += np.sign(steps[max_index]) * (-1)
                    else:
                        config_new.ee2_angles[max_index] = config2.ee2_angles[max_index]
                        # update the steps
                        steps[max_index] = 0
                # change the lengths
                else:
                    if np.abs(steps[max_index]) > 1:
                        config_new.lengths[max_index - self.num_segments] += np.sign(steps[max_index]) * (
                            -1) * length_preminum
                        # update the steps
                        steps[max_index] += np.sign(steps[max_index]) * (-1)
                    else:
                        config_new.lengths[max_index - self.num_segments] = config2.lengths[
                            max_index - self.num_segments]
                        # update the steps
                        steps[max_index] = 0

                # update the points
                config_new = RobotConfig(config_new.lengths, ee2x=config_new.get_ee1()[0], ee2y=config_new.get_ee1()[1],
                                         ee2_angles=config_new.ee1_angles,
                                         ee2_grappled=True)

            # update the max_index
            max_index = list(np.abs(steps)).index(max(np.abs(steps)))

            # update the result_list
            config_record = copy.deepcopy(config_new)
            result_list.append(config_record)
        return result_list

    def inter_grapple_point_check(self):
        # new
        """check whether there is an inter grapple point."""
        grapple_num = len(self.problem_spec.grapple_points)
        return grapple_num > 2

    def inter_grapple_point(self):
        # new
        """return the inter grapple point."""
        grapple_point = self.problem_spec.grapple_points[1]
        return grapple_point

    def how_to_choose_path(self):
        """choose a path from which config to another config."""
        # if grapple_num is 2 and self.ee1_grappled are same
        path = []
        if not (self.inter_grapple_point_check()) and (self.initial.ee1_grappled == self.goal.ee1_grappled):
            path = [(self.initial, self.goal)]

        # if grapple_num is 2 and self.ee1_grappled are different
        if not (self.inter_grapple_point_check()) and (self.initial.ee1_grappled != self.goal.ee1_grappled):
            print("goal_point[0]:", self.goal.points[0])
            print("goal_point[-1]:", self.goal.points[-1])
            if self.goal.ee1_grappled:
                inter_config = self.different_ee_goal_generator_success(self.initial, self.goal.get_ee1())
            else:
                inter_config = self.different_ee_goal_generator_success(self.initial, self.goal.get_ee2())
            inter_config_reverse = self.change_config_direction(inter_config)  # error
            path = [(self.initial, inter_config), (inter_config_reverse, self.goal)]

        # if grapple_num is larger than 2
        if self.inter_grapple_point_check():
            # 1->2
            # 1: self.initial
            # 2: inter_config_1
            inter_config_1 = self.different_ee_goal_generator_success(self.initial, self.inter_grapple_point())
            path.append([self.initial, inter_config_1])
            # print("1:",self.initial.ee1_grappled)
            # print("2:", inter_config_1.ee1_grappled)

            # 3 -> 4
            inter_config_1_reverse = self.change_config_direction(inter_config_1)  # 3  -> ee2
            print("3:", inter_config_1_reverse.ee2_grappled)
            print("goal_ee1:", self.goal.ee1_grappled)
            print("goal_ee2:", self.goal.ee2_grappled)
            print("goal:", self.goal.get_ee1())
            print("inter_config_1_reverse:", inter_config_1_reverse)
            # gygygy
            if self.goal.ee1_grappled:
                inter_config_2 = self.different_ee_goal_generator_success(inter_config_1_reverse, self.goal.get_ee1())
                print("111")
            else:
                inter_config_2 = self.different_ee

        return path

    def output(self, config_list):
        # new
        """output the result as a list of RobotConfig. At the same time, print it out on console.
        input: a list of configs [config1, config2, config3] (config is on the graph)
        output : [RobotConfig, RobotConfig, RobotConfig,...]"""
        all_result = []
        for i in range(len(config_list) - 1):
            result_list = self.collision_free_steps_generator(config_list[i], config_list[i + 1])
            all_result.extend(result_list)
            print(i, ":", len(all_result))
        for j in range(len(all_result)):
            print(all_result[j], "\n")
        return all_result

    def output_path(self, config_list):
        # new
        """output the result as a list of RobotConfig. At the same time, print it out on console.
        input: a list of configs [config1, config2, config3] (config is on the graph)
        output : [RobotConfig, RobotConfig, RobotConfig,...]"""
        for i in range(len(config_list)):
            print(config_list[i], "\n")

    def output_to_txt(self, config_list, file):
        # new
        """output the result as a list of RobotConfig. At the same time, print it out on console.
        input: a list of configs [config1, config2, config3] (config is on the graph)
        output : [RobotConfig, RobotConfig, RobotConfig,...]
        output_file :  output.txt """
        all_result = []
        for i in range(len(config_list) - 1):
            result_list = self.collision_free_steps_generator(config_list[i], config_list[i + 1])
            all_result.extend(result_list)
        write_robot_config_list_to_file(file, all_result)

    def output_path_to_txt(self, config_list, file):
        # new
        """output the result as a list of RobotConfig. At the same time, print it out on console.
        input: a list of configs [config1, config2, config3] (config is on the graph)
        output : [RobotConfig, RobotConfig, RobotConfig,...]
        output_file :  output.txt """
        write_robot_config_list_to_file(file, config_list)

def main(arglist):
    run_prm(arglist)

    # run_rrt(arglist)

def run_rrt(arglist):
    solver = Solver(arglist[0])
    if len(arglist) > 1:
        output_file = arglist[1]
    initial_config = solver.problem_spec.initial
    goal_config = solver.problem_spec.goal
    initial_graph = {}
    init_bunch_configs = {}
    init_distance_from_config = {}
    solution_found = False

    while not solution_found:
        print("RRT start at", datetime.datetime.now())
        graph, vertex_count,  solution_found = solver.rrt(initial_config, goal_config, initial_graph, 50, 0.3)
        initial_graph = graph

    print("Solution found at", datetime.datetime.now())
    if solution_found and len(solver.config_list) > 2:
        print("BFS start at", datetime.datetime.now())
        solver.config_list, solution_found = solver.BFS(graph)
        # solver.output_to_file(output_file)
        for config in solver.config_list:
            print(config)
    else:
        print("Solution not found")

def run_prm(arglist):

    solver = Solver(arglist[0])
    if len(arglist) > 1:
        output_file = arglist[1]
    initial_config = solver.problem_spec.initial
    goal_config = solver.problem_spec.goal
    initial_graph = {}
    init_bunch_configs = {}
    init_distance_from_config = {}
    solution_found = False


    path = solver.how_to_choose_path()

    for i in range(len(path)):

        print("Leg", i + 1)
        initial_graph = {}
        init_bunch_configs = {}
        init_distance_from_config = {}
        config_list = []
        solution_found = False

        while not solution_found:
            print("PRM start at", datetime.datetime.now())
            graph = solver.prm(path[i][0], path[i][1], initial_graph, init_distance_from_config, 10, 1, 0.3)
            print("BFS start at", datetime.datetime.now())
            # print("RRT start at", datetime.datetime.now())
            # graph, vertex_count, solution_found = solver.rrt(path[i][0], path[i][1], initial_graph, 50, 0.3)
            # print("BFS start at", datetime.datetime.now())

            config_list, solution_found = solver.BFS(graph)
            if not solution_found:
                initial_graph = graph
                # init_distance_from_config = distance_from_config

        print("Solution found for leg {0} at {1}".format(i + 1, str(datetime.datetime.now())))
        solver.config_list.extend(config_list)

    print("Solution found at", datetime.datetime.now())
    # found eventually for every leg
    if solution_found:
        if len(arglist) > 1:
            solver.output_to_txt(solver.config_list, arglist[1])
            # solver.output_path_to_txt(solver.config_list, arglist[1])
        else:
            # solver.output(solver.config_list)
            solver.output_path(solver.config_list)
    else:
        print("Solution not found")



if __name__ == '__main__':
    main(sys.argv[1:])
