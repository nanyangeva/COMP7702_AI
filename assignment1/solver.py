import sys
import msvcrt
import numpy as np
import time
import operator
import copy

class SokobanMap:
    # input file symbols
    BOX_SYMBOL = 'B'
    TGT_SYMBOL = 'T'
    PLAYER_SYMBOL = 'P'
    OBSTACLE_SYMBOL = '#'
    FREE_SPACE_SYMBOL = ' '
    BOX_ON_TGT_SYMBOL = 'b'
    PLAYER_ON_TGT_SYMBOL = 'p'

    # move symbols (i.e. output file symbols)
    LEFT = 'l'
    RIGHT = 'r'
    UP = 'u'
    DOWN = 'd'

    # render characters
    FREE_GLYPH = '   '
    OBST_GLYPH = 'XXX'
    BOX_GLYPH = '[B]'
    TGT_GLYPH = '(T)'
    PLAYER_GLYPH = '<P>'

class SokobanMapMod:
    """
    Instance of a Sokoban game map. You may use this class and its functions
    directly or duplicate and modify it in your solution. You should avoid
    modifying this file directly.
    COMP3702 2019 Assignment 1 Support Code
    Last updated by njc 11/08/19
    """

    def __init__(self, filename):
        """
        Build a Sokoban map instance from the given file name
        :param filename:
        """
        f = open(filename, 'r')

        rows = []
        for line in f:
            if len(line.strip()) > 0:
                rows.append(list(line.strip()))

        f.close()

        row_len = len(rows[0])
        for row in rows:
            assert len(row) == row_len, "Mismatch in row length"

        num_rows = len(rows)

        box_positions = []
        tgt_positions = []
        player_position = None
        for i in range(num_rows):
            for j in range(row_len):
                if rows[i][j] == SokobanMap.BOX_SYMBOL:
                    box_positions.append((i, j))
                    rows[i][j] = SokobanMap.FREE_SPACE_SYMBOL
                elif rows[i][j] == SokobanMap.TGT_SYMBOL:
                    tgt_positions.append((i, j))
                    rows[i][j] = SokobanMap.FREE_SPACE_SYMBOL
                elif rows[i][j] == SokobanMap.PLAYER_SYMBOL:
                    player_position = (i, j)
                    rows[i][j] = SokobanMap.FREE_SPACE_SYMBOL
                elif rows[i][j] == SokobanMap.BOX_ON_TGT_SYMBOL:
                    box_positions.append((i, j))
                    tgt_positions.append((i, j))
                    rows[i][j] = SokobanMap.FREE_SPACE_SYMBOL
                elif rows[i][j] == SokobanMap.PLAYER_ON_TGT_SYMBOL:
                    player_position = (i, j)
                    tgt_positions.append((i, j))
                    rows[i][j] = SokobanMap.FREE_SPACE_SYMBOL

        assert len(box_positions) == len(tgt_positions), "Number of boxes does not match number of targets"

        self.x_size = row_len
        self.y_size = num_rows
        self.box_positions = box_positions
        self.tgt_positions = tgt_positions
        self.player_position = player_position
        self.player_x = player_position[1]
        self.player_y = player_position[0]
        self.obstacle_map = rows
        self.by_move = ''

    def get_state_tuple(self):
        state_tuple = []
        sorted_box_positions = self.box_positions
        sorted_box_positions.sort(key = operator.itemgetter(0,1))
        for box_position in sorted_box_positions:
            state_tuple.append(box_position[0])
            state_tuple.append(box_position[1])
        state_tuple.append(self.player_y)
        state_tuple.append(self.player_x)
        return tuple(state_tuple)

    def apply_move(self, move):
        """
        Apply a player move to the map.
        :param move: 'L', 'R', 'U' or 'D'
        :return: return self if move was successful, None if move could not be completed
        """
        # prepare a new state instance
        new_state = copy.deepcopy(self)
        # basic obstacle check
        if move == SokobanMap.LEFT:
            if self.obstacle_map[self.player_y][self.player_x - 1] == SokobanMap.OBSTACLE_SYMBOL:
                return None
            else:
                new_x = self.player_x - 1
                new_y = self.player_y

        elif move == SokobanMap.RIGHT:
            if self.obstacle_map[self.player_y][self.player_x + 1] == SokobanMap.OBSTACLE_SYMBOL:
                return None
            else:
                new_x = self.player_x + 1
                new_y = self.player_y

        elif move == SokobanMap.UP:
            if self.obstacle_map[self.player_y - 1][self.player_x] == SokobanMap.OBSTACLE_SYMBOL:
                return None
            else:
                new_x = self.player_x
                new_y = self.player_y - 1

        else:
            if self.obstacle_map[self.player_y + 1][self.player_x] == SokobanMap.OBSTACLE_SYMBOL:
                return None
            else:
                new_x = self.player_x
                new_y = self.player_y + 1

        # pushed box collision check
        if (new_y, new_x) in self.box_positions:
            if move == SokobanMap.LEFT:
                # box hit the obstacle if pushed or box hit any another box if pushed
                if self.obstacle_map[new_y][new_x - 1] == SokobanMap.OBSTACLE_SYMBOL or (
                        new_y, new_x - 1) in self.box_positions:
                    return None
                else:
                    new_box_x = new_x - 1
                    new_box_y = new_y

            elif move == SokobanMap.RIGHT:
                if self.obstacle_map[new_y][new_x + 1] == SokobanMap.OBSTACLE_SYMBOL or (
                        new_y, new_x + 1) in self.box_positions:
                    return None
                else:
                    new_box_x = new_x + 1
                    new_box_y = new_y

            elif move == SokobanMap.UP:
                if self.obstacle_map[new_y - 1][new_x] == SokobanMap.OBSTACLE_SYMBOL or (
                        new_y - 1, new_x) in self.box_positions:
                    return None
                else:
                    new_box_x = new_x
                    new_box_y = new_y - 1

            else:
                if self.obstacle_map[new_y + 1][new_x] == SokobanMap.OBSTACLE_SYMBOL or (
                        new_y + 1, new_x) in self.box_positions:
                    return None
                else:
                    new_box_x = new_x
                    new_box_y = new_y + 1

            # update box position
            new_state.box_positions.remove((new_y, new_x))
            new_state.box_positions.append((new_box_y, new_box_x))

        # update player position
        new_state.player_x = new_x
        new_state.player_y = new_y
        new_state.player_position = (new_y, new_x)
        new_state.by_move = move

        rows = self.render(new_state)
        for i in range(new_state.y_size):
            for j in range(new_state.x_size):
                if rows[i][j] == SokobanMap.BOX_SYMBOL:
                    rows[i][j] = SokobanMap.FREE_SPACE_SYMBOL
                elif rows[i][j] == SokobanMap.TGT_SYMBOL:
                    rows[i][j] = SokobanMap.FREE_SPACE_SYMBOL
                elif rows[i][j] == SokobanMap.PLAYER_SYMBOL:
                    rows[i][j] = SokobanMap.FREE_SPACE_SYMBOL
                elif rows[i][j] == SokobanMap.BOX_ON_TGT_SYMBOL:
                    rows[i][j] = SokobanMap.FREE_SPACE_SYMBOL
                elif rows[i][j] == SokobanMap.PLAYER_ON_TGT_SYMBOL:
                    rows[i][j] = SokobanMap.FREE_SPACE_SYMBOL
        new_state.obstacle_map = rows

        return new_state

    def explore_apply_move(self, move):
        # basic obstacle check
        if move == SokobanMap.LEFT:
            if self.obstacle_map[self.player_y][self.player_x - 1] == SokobanMap.OBSTACLE_SYMBOL:
                return False
            else:
                new_x = self.player_x - 1
                new_y = self.player_y

        elif move == SokobanMap.RIGHT:
            if self.obstacle_map[self.player_y][self.player_x + 1] == SokobanMap.OBSTACLE_SYMBOL:
                return False
            else:
                new_x = self.player_x + 1
                new_y = self.player_y

        elif move == SokobanMap.UP:
            if self.obstacle_map[self.player_y - 1][self.player_x] == SokobanMap.OBSTACLE_SYMBOL:
                return False
            else:
                new_x = self.player_x
                new_y = self.player_y - 1

        else:
            if self.obstacle_map[self.player_y + 1][self.player_x] == SokobanMap.OBSTACLE_SYMBOL:
                return False
            else:
                new_x = self.player_x
                new_y = self.player_y + 1

        # pushed box collision check(lock check)
        if (new_y, new_x) in self.box_positions:
            if move == SokobanMap.LEFT:
                # box hit the obstacle if pushed or box hit any another box if pushed
                if self.obstacle_map[new_y][new_x - 1] == SokobanMap.OBSTACLE_SYMBOL or (
                        new_y, new_x - 1) in self.box_positions:
                    return False
                else:
                    if self.is_beside_obstacle(new_x, new_y,
                                               move):  # positioned by the left obstacle after box has been pushed
                        # if happens to be the target
                        is_pushed_to_target = False
                        for i in range(len(self.tgt_positions)):
                            if self.tgt_positions[i][1] == new_x - 1 and self.tgt_positions[i][0] == new_y:
                                new_box_x = new_x - 1
                                new_box_y = new_y
                                return True
                        if is_pushed_to_target == True:
                            return True
                        # if get stuck in corner
                        if self.is_stuck_in_corner(new_x, new_y, move):
                            return False
                        # if get stuck in trap
                        if self.is_stuck_in_trap(new_x, new_y, move):
                            return False
                            # return self.has_target_in_trap(new_x, new_y, move)
                        # # if is beside boundary
                        # if self.is_beside_boundary(new_x, new_y, move):
                        #     for i in range(len(self.tgt_positions)):
                        #         if self.tgt_positions[i][1] != new_x - 1 or (
                        #                 self.tgt_positions[i][0] <= new_y and new_y == self.y_size - 1):
                        #             return False
                        #         if self.tgt_positions[i][1] != new_x + 1 or (
                        #                 self.tgt_positions[i][0] >= new_y and new_y == 1):
                        #             return False
                        else:
                            return True

                    new_box_x = new_x - 1
                    new_box_y = new_y

            elif move == SokobanMap.RIGHT:
                if self.obstacle_map[new_y][new_x + 1] == SokobanMap.OBSTACLE_SYMBOL or (
                        new_y, new_x + 1) in self.box_positions:
                    return False
                else:
                    if self.is_beside_obstacle(new_x, new_y,
                                               move):  # positioned by the right obstacle after box has been pushed
                        # if happens to be the target
                        is_pushed_to_target = False
                        for i in range(len(self.tgt_positions)):
                            if self.tgt_positions[i][1] == new_x + 1 and self.tgt_positions[i][0] == new_y:
                                new_box_x = new_x + 1
                                new_box_y = new_y
                                return True
                        if is_pushed_to_target == True:
                            return True
                        # if get stuck in corner
                        if self.is_stuck_in_corner(new_x, new_y, move):
                            return False
                        # if get stuck in trap
                        if self.is_stuck_in_trap(new_x, new_y, move):
                            return False
                            # return self.has_target_in_trap(new_x, new_y, move)
                        # if is beside boundary
                        # if self.is_beside_boundary(new_x, new_y, move):
                        #     for i in range(len(self.tgt_positions)):
                        #         if self.tgt_positions[i][1] != new_x + 1 or (
                        #                 self.tgt_positions[i][0] <= new_y and new_y == self.y_size - 1):
                        #             return False
                        #         if self.tgt_positions[i][1] != new_x + 1 or (
                        #                 self.tgt_positions[i][0] >= new_y and new_y == 1):
                        #             return False
                        else:
                            return True
                    new_box_x = new_x + 1
                    new_box_y = new_y

            elif move == SokobanMap.UP:
                if self.obstacle_map[new_y - 1][new_x] == SokobanMap.OBSTACLE_SYMBOL or (
                        new_y - 1, new_x) in self.box_positions:
                    return False
                else:
                    if self.is_beside_obstacle(new_x, new_y,
                                               move):  # positioned by the up obstacle after box has been pushed
                        # if happens to be the target
                        is_pushed_to_target = False
                        for i in range(len(self.tgt_positions)):
                            if self.tgt_positions[i][0] == new_y - 1 and self.tgt_positions[i][1] == new_x:
                                new_x = self.player_x
                                new_y = self.player_y - 1
                                return True
                        if is_pushed_to_target == True:
                            return True
                        # if get stuck in corner
                        if self.is_stuck_in_corner(new_x, new_y, move):
                            return False
                        # if get stuck in trap
                        if self.is_stuck_in_trap(new_x, new_y, move):
                            return False
                            # return self.has_target_in_trap(new_x, new_y, move)
                        # if is beside boundary
                        # if self.is_beside_boundary(new_x, new_y, move):
                        #     for i in range(len(self.tgt_positions)):
                        #         if self.tgt_positions[i][0] != new_y - 1 or (
                        #                 self.tgt_positions[i][1] <= new_x and new_x == self.x_size - 1):
                        #             return False
                        #         if self.tgt_positions[i][0] != new_y - 1 or (
                        #                 self.tgt_positions[i][1] >= new_x and new_x == 1):
                        #             return False
                        else:
                            return True
                    new_box_x = new_x
                    new_box_y = new_y - 1

            else:
                if self.obstacle_map[new_y + 1][new_x] == SokobanMap.OBSTACLE_SYMBOL or (
                        new_y + 1, new_x) in self.box_positions:
                    return False
                else:
                    if self.is_beside_obstacle(new_x, new_y,
                                               move):  # positioned by the left obstacle after box has been pushed
                        # if happens to be the target
                        is_pushed_to_target = False
                        for i in range(len(self.tgt_positions)):
                            if self.tgt_positions[i][0] == new_y + 1 and self.tgt_positions[i][1] == new_x:
                                new_x = self.player_x
                                new_y = self.player_y + 1
                                is_pushed_to_target = True
                        if is_pushed_to_target == True:
                            return True
                        # if get stuck in corner
                        if self.is_stuck_in_corner(new_x, new_y, move):
                            return False
                        # if get stuck in trap
                        if self.is_stuck_in_trap(new_x, new_y, move):
                            return False
                            # return self.has_target_in_trap(new_x, new_y, move)
                        # if is beside boundary
                        # if self.is_beside_boundary(new_x, new_y, move):
                        #     for i in range(len(self.tgt_positions)):
                        #         if self.tgt_positions[i][0] != new_y + 1 or (
                        #                 self.tgt_positions[i][1] <= new_x and new_x == self.x_size - 1):
                        #             return False
                        #         if self.tgt_positions[i][0] != new_y + 1 or (
                        #                 self.tgt_positions[i][1] >= new_x and new_x == 1):
                        #             return False
                        else:
                            return True
                    new_box_x = new_x
                    new_box_y = new_y + 1

        return True

    def set_map(self, box_positions, tgt_positions, player_position, obstacle_map):
        self.box_positions = box_positions
        self.tgt_positions = tgt_positions
        self.player_position = player_position
        self.obstacle_map = obstacle_map

    def render(self, state):
        rows = []
        for r in range(state.y_size):
            line = []
            for c in range(state.x_size):
                if state.obstacle_map[r][c] == SokobanMap.OBSTACLE_SYMBOL:
                    symbol = SokobanMap.OBSTACLE_SYMBOL
                else:
                    if state.obstacle_map[r][c] == SokobanMap.FREE_SPACE_SYMBOL:
                        symbol = ' '
                        if (r, c) in state.box_positions and (r, c) in state.tgt_positions:
                            symbol = SokobanMap.BOX_ON_TGT_SYMBOL
                            # self.obstacle_map[r][c] = 'b'
                        elif (r, c) in state.tgt_positions and (r, c) == (state.player_y, state.player_x):
                            self.obstacle_map[r][c] = SokobanMap.PLAYER_ON_TGT_SYMBOL
                        elif (r, c) in state.tgt_positions:
                            symbol = 'T'
                        # box or player overwrites tgt
                        elif (r, c) in state.box_positions:
                            symbol = 'B'
                        elif state.player_x == c and state.player_y == r:
                            symbol = 'P'
                        else:
                            symbol = ' '

                line.append(symbol)
            rows.append(line)
        return rows

    def calculate_heuristic_cost(self):
        # find the closest box from the player
        player_to_box_distances = []
        box_in_tgt_indexes = []
        box_index = 0
        for box_position in self.box_positions:
            for tgt_position in self.tgt_positions:
                if box_position[0] == tgt_position[0] and box_position[1] == tgt_position[1]:
                    box_in_tgt_indexes.append(box_index)
                    break

            player_to_box_distances.append(abs(box_position[0] - self.player_y) + abs(box_position[1] - self.player_x))
            box_index = box_index + 1


        # calculate the closest box's distances to each target and pick the smallest one
        if len(box_in_tgt_indexes) == len(self.box_positions):
            return 0

        index = 0
        min_distance = player_to_box_distances[0]
        min_ind = 0
        for player_to_box_distance in player_to_box_distances:
            if player_to_box_distance < min_distance and not index in box_in_tgt_indexes:
                min_distance = player_to_box_distance
                min_ind = index
            index = index + 1

        box_position = self.box_positions[min_ind]
        box_to_tgt_distances = []
        for tgt_position in self.tgt_positions:
            is_tgt_occupied = False
            for box_position in self.box_positions:
                if box_position[0] == tgt_position[0] and box_position[1] == tgt_position[1]:
                    is_tgt_occupied = True
                    break
            if not is_tgt_occupied:
                box_to_tgt_distances.append(abs(tgt_position[0] - box_position[0]) + abs(tgt_position[1] - box_position[1]))
        heuristic_cost = min(player_to_box_distances) + min(box_to_tgt_distances)
        return heuristic_cost

    def is_beside_obstacle(self, new_x, new_y, move):
        if move == SokobanMap.LEFT:
            if new_x - 2 >= 0:
                is_obstacle = self.obstacle_map[new_y][new_x - 2] == SokobanMap.OBSTACLE_SYMBOL
                return is_obstacle
            else:
                return False
        elif move == SokobanMap.RIGHT:
            if new_x + 2 <= self.x_size - 1:
                is_obstacle = self.obstacle_map[new_y][new_x + 2] == SokobanMap.OBSTACLE_SYMBOL
                return is_obstacle
            else:
                return False
        elif move == SokobanMap.UP:
            if new_y - 2 >= 0:
                is_obstacle = self.obstacle_map[new_y - 2][new_x] == SokobanMap.OBSTACLE_SYMBOL
                return is_obstacle
            else:
                return False
        else:
            if new_y + 2 <= self.y_size - 1:
                is_obstacle = self.obstacle_map[new_y + 2][new_x] == SokobanMap.OBSTACLE_SYMBOL
                return is_obstacle
            else:
                return False

    # stuck in corner
    ####B               #####
    #########################
    def is_stuck_in_corner(self, new_x, new_y, move):
        if move == SokobanMap.LEFT:
            # check up
            if new_y - 1 >= 0:
                if self.obstacle_map[new_y - 1][new_x - 1] == SokobanMap.OBSTACLE_SYMBOL:
                    return True
                for box_position in self.box_positions:
                    if box_position[0] == new_y - 1 and box_position[1] == new_x - 1:
                        return True
            # check down
            if new_y + 1 <= self.y_size - 1:
                if self.obstacle_map[new_y + 1][new_x - 1] == SokobanMap.OBSTACLE_SYMBOL:
                    return True
                for box_position in self.box_positions:
                    if box_position[0] == new_y + 1 and box_position[1] == new_x - 1:
                        return True
            return False
        elif move == SokobanMap.RIGHT:
            # check up
            if new_y - 1 >= 0:
                if self.obstacle_map[new_y - 1][new_x + 1] == SokobanMap.OBSTACLE_SYMBOL:
                    return True
                for box_position in self.box_positions:
                    if box_position[0] == new_y - 1 and box_position[1] == new_x + 1:
                        return True
            # check down
            if new_y + 1 <= self.y_size - 1:
                if self.obstacle_map[new_y + 1][new_x + 1] == SokobanMap.OBSTACLE_SYMBOL:
                    return True
                for box_position in self.box_positions:
                    if box_position[0] == new_y + 1 and box_position[1] == new_x + 1:
                        return True
            return False
        elif move == SokobanMap.UP:
            # check left
            if new_x - 1 >= 0:
                if self.obstacle_map[new_y - 1][new_x - 1] == SokobanMap.OBSTACLE_SYMBOL:
                    return True
                for box_position in self.box_positions:
                    if box_position[0] == new_y - 1 and box_position[1] == new_x - 1:
                        return True
            # check right
            if new_x + 1 >= self.x_size - 1:
                if self.obstacle_map[new_y - 1][new_x + 1] == SokobanMap.OBSTACLE_SYMBOL:
                    return True
                for box_position in self.box_positions:
                    if box_position[0] == new_y - 1 and box_position[1] == new_x + 1:
                        return True
            return False
        elif move == SokobanMap.DOWN:
            # check left
            if new_x - 1 >= 0:
                if self.obstacle_map[new_y + 1][new_x - 1] == SokobanMap.OBSTACLE_SYMBOL:
                    return True
                for box_position in self.box_positions:
                    if box_position[0] == new_y + 1 and box_position[1] == new_x - 1:
                        return True
            # check right
            if new_x + 1 >= self.x_size - 1:
                if self.obstacle_map[new_y + 1][new_x + 1] == SokobanMap.OBSTACLE_SYMBOL:
                    return True
                for box_position in self.box_positions:
                    if box_position[0] == new_y + 1 and box_position[1] == new_x + 1:
                        return True
            return False
        else:
            return False

    # trap
    ####        B       #####
    #########################
    def is_stuck_in_trap(self, new_x, new_y, move):
        if move == SokobanMap.LEFT or move == SokobanMap.RIGHT:
            if self.find_nearest_u_gap(new_x, new_y, move) > self.find_nearest_u_obstacle(new_x, new_y, move) or self.find_nearest_d_gap(new_x, new_y, move) < self.find_nearest_d_obstacle(new_x, new_y, move):
                return False
            elif self.find_nearest_u_tgt(new_x, new_y, move) > self.find_nearest_u_obstacle(new_x, new_y, move) or self.find_nearest_d_tgt(new_x, new_y, move) < self.find_nearest_d_obstacle(new_x, new_y, move):
                return False
            else:
                return True
        else:
            if self.find_nearest_l_gap(new_x, new_y, move) > self.find_nearest_l_obstacle(new_x, new_y, move) or self.find_nearest_r_gap(new_x, new_y, move) < self.find_nearest_r_obstacle(new_x, new_y, move):
                return False
            elif self.find_nearest_l_tgt(new_x, new_y, move) > self.find_nearest_l_obstacle(new_x, new_y, move) or self.find_nearest_r_tgt(new_x, new_y, move) < self.find_nearest_r_obstacle(new_x, new_y, move):
                return False
            else:
                return True

    def get_gap_offset(self, move):
        offset = 0
        if move == SokobanMap.LEFT or move == SokobanMap.UP:
            offset = -2
        if move == SokobanMap.RIGHT or move == SokobanMap.DOWN:
            offset = 2
        return offset

    def get_box_offset(self, move):
        offset = 0
        if move == SokobanMap.LEFT or move == SokobanMap.UP:
            offset = -1
        if move == SokobanMap.RIGHT or move == SokobanMap.DOWN:
            offset = 1
        return offset

    def find_nearest_u_tgt(self, new_x, new_y, move):
        offset = self.get_box_offset(move)
        while new_y >= 1:
            if not (new_y, new_x + offset) in self.tgt_positions:
                new_y = new_y - 1
            else:
                return new_y
        return  -1

    def find_nearest_d_tgt(self, new_x, new_y, move):
        offset = self.get_box_offset(move)
        while new_y <= self.y_size-1:
            if not (new_y, new_x + offset) in self.tgt_positions:
                new_y = new_y + 1
            else:
                return new_y
        return 99999

    def find_nearest_l_tgt(self, new_x, new_y, move):
        offset = self.get_box_offset(move)
        while new_x >= 1:
            if not (new_y + offset, new_x) in self.tgt_positions:
                new_x = new_x - 1
            else:
                return new_x
        return  -1

    def find_nearest_r_tgt(self, new_x, new_y, move):
        offset = self.get_box_offset(move)
        while new_x <= self.x_size-1:
            if not (new_y + offset, new_x) in self.tgt_positions:
                new_x = new_x + 1
            else:
                return new_x
        return 99999

    def find_nearest_u_gap(self, new_x, new_y, move):
        offset = self.get_gap_offset(move)
        while new_y >= 1:
            if self.obstacle_map[new_y][new_x + offset] != SokobanMap.FREE_SPACE_SYMBOL:
                new_y = new_y - 1
            else:
                return new_y
        return  -1

    def find_nearest_d_gap(self, new_x, new_y, move):
        offset = self.get_gap_offset(move)
        while new_y <= self.y_size-1:
            if self.obstacle_map[new_y][new_x + offset] != SokobanMap.FREE_SPACE_SYMBOL:
                new_y = new_y + 1
            else:
                return new_y
        return 99999

    def find_nearest_l_gap(self, new_x, new_y, move):
        offset = self.get_gap_offset(move)
        while new_x >= 1:
            if self.obstacle_map[new_y + offset][new_x] != SokobanMap.FREE_SPACE_SYMBOL:
                new_x = new_x - 1
            else:
                return new_x
        return  -1

    def find_nearest_r_gap(self, new_x, new_y, move):
        offset = self.get_gap_offset(move)
        while new_x <= self.x_size-1:
            if self.obstacle_map[new_y + offset][new_x] != SokobanMap.FREE_SPACE_SYMBOL:
                new_x = new_x + 1
            else:
                return new_x
        return 99999

    def find_nearest_u_obstacle(self, new_x, new_y, move):
        offset = self.get_box_offset(move)
        while new_y >= 1:
            if self.obstacle_map[new_y][new_x + offset] != SokobanMap.OBSTACLE_SYMBOL:
                new_y = new_y - 1
            else:
                return new_y
        return  -1

    def find_nearest_d_obstacle(self, new_x, new_y, move):
        offset = self.get_box_offset(move)
        while new_y <= self.y_size-1:
            if self.obstacle_map[new_y][new_x + offset] != SokobanMap.OBSTACLE_SYMBOL:
                new_y = new_y + 1
            else:
                return new_y
        return 99999

    def find_nearest_l_obstacle(self, new_x, new_y, move):
        offset = self.get_box_offset(move)
        while new_x >= 1:
            if self.obstacle_map[new_y + offset][new_x] != SokobanMap.OBSTACLE_SYMBOL:
                new_x = new_x - 1
            else:
                return new_x
        return  -1

    def find_nearest_r_obstacle(self, new_x, new_y, move):
        offset = self.get_box_offset(move)
        while new_x <= self.x_size-1:
            if self.obstacle_map[new_y + offset][new_x] != SokobanMap.OBSTACLE_SYMBOL:
                new_x = new_x + 1
            else:
                return new_x
        return 99999

    def has_target_in_trap(self, new_x, new_y, move):
        offset = self.get_box_offset(move)
        for i in range(len(self.tgt_positions)):
            if move == SokobanMap.LEFT or move == SokobanMap.RIGHT:
                if self.tgt_positions[i][1] == new_x + offset:
                    return True
            if move == SokobanMap.UP or move == SokobanMap.DOWN:
                if self.tgt_positions[i][0] == new_y + offset:
                    return True

        return False

    def is_beside_boundary(self, new_x, new_y, move):
        if move == SokobanMap.LEFT:
            if new_x - 2 == 0:
                return self.obstacle_map[new_y][new_x - 2] == SokobanMap.OBSTACLE_SYMBOL
            else:
                return False
        elif move == SokobanMap.RIGHT:
            if new_x + 2 == self.x_size - 1:
                return self.obstacle_map[new_y][new_x + 2] == SokobanMap.OBSTACLE_SYMBOL
            else:
                return False
        elif move == SokobanMap.UP:
            if new_y - 2 == 0:
                return self.obstacle_map[new_y - 2][new_x] == SokobanMap.OBSTACLE_SYMBOL
            else:
                return False
        else:
            if new_y + 2 == self.y_size - 1:
                return self.obstacle_map[new_y + 2][new_x] == SokobanMap.OBSTACLE_SYMBOL
            else:
                return False

    def is_finished(self):
        finished = True
        for i in self.box_positions:
            if i not in self.tgt_positions:
                finished = False
        return finished

# BFS node
class NodeMod():

    def __init__(self, state:SokobanMapMod, parent=None, action=None, depth=0):
        self.state = state  # a board
        self.parent = parent  # parent node, a NODE! not just a matrix.
        self.action = action  # The one that led to this node (useful for retracing purpose)
        self.depth = depth  # depth of the node in the tree. This is the criterion for who's next in DFS, BFS.

    def state_exists_in_frontier_states(self, state:SokobanMapMod, frontier):
        return state.get_state_tuple() in frontier

    def state_exists_in_explorered_states(self, state, explored):
        return state.get_state_tuple() in explored

    def world_dynamics(self, current_state, action):
        if action == SokobanMap.UP:
            new_state = current_state.apply_move(SokobanMap.UP)
            if new_state is None:
                raise Exception('Inappropriate action for the current state')
        elif action == SokobanMap.DOWN:
            new_state =  current_state.apply_move(SokobanMap.DOWN)
            if new_state is None:
                raise Exception('Inappropriate action for the current state')
        elif action == SokobanMap.LEFT:
            new_state = current_state.apply_move(SokobanMap.LEFT)
            if new_state is None:
                raise Exception('Inappropriate action for the current state')
        elif action == SokobanMap.RIGHT:
            new_state = current_state.apply_move(SokobanMap.RIGHT)
            if new_state is None:
                raise Exception('Inappropriate action for the current state')
        else:
            print('Unknown action!')
        return new_state

    def explore_actions(self, state):
        possibilities = [SokobanMap.LEFT, SokobanMap.RIGHT, SokobanMap.UP, SokobanMap.DOWN]
        actions = []
        for apossibility in possibilities:
                is_possible = state.explore_apply_move(apossibility)
                if is_possible == True:
                    actions.append(apossibility)
        return actions

    def whos_next_BFS(self, frontier):
        least_depth = 99999999999
        least_depth_node_index = None
        for index, anode in enumerate(frontier):
            if anode.depth < least_depth:
                least_depth = anode.depth
                least_depth_node_index = index
        return least_depth_node_index

    def done(self, current_node):
        """ The prupose of this function  is: Trace back this node to the founding granpa.
        Print out the states through out
        """
        solution = []
        founding_father = current_node
        states = []  # the retraced states will be stored here.
        counter = 0
        limit = 50  # if the trace is longer than 50, don't print anything, it will be a mess.
        while founding_father:
            states.append(founding_father.state)
            founding_father = founding_father.parent
            counter += 1
            # Keep doing this until you reach the founding father that has a parent None (see default of init method)
        print('Number of steps to the goal = ', counter)
        if counter > limit:
            print('Too many steps to be printed')
        else:
            for i in reversed(states):  # Cause we want to print solution from initial to goal not the opposite.
                print('By Move:' + i.by_move + '\t' + 'Player Position:(' + str(i.player_y) + ',' + str(i.player_x)+ ')', '\n')
                if i.by_move != '':
                    solution.append(i.by_move)

            return solution

    def goal_test(self):
        return self.state.is_finished()

    def add_to_explored_states(self, node, explored):
        explored.add(node.state.get_state_tuple())

    def BFS(self):
        start = time.time()
        frontier = [self]  # queue of found but unvisited nodes, FIFO
        frontier_max_size = len(frontier)  # We measure space / memory requirements of the algo via this metric.
        ft = set()
        ft.add(self.state.get_state_tuple())
        # A version of the frontier that contains only states, not nodes
        # This is version is required because Python is slow, and having this numeric version
        # will make things much faster.

        explored = set([])
        # We use this to measure time requirements (#visited_nodes). This is unbiased measure and doesn't depend on computer speed
        # We should have another set of unexplored, but it is huge and we will ignore it.
        # The union of the three sets above is the state space.

        # Let's start exploring the frontier.
        ct = 0  # a counter to see the progress.
        while frontier:
            '''
            As long as there are elements in the frontier, then the search is on. But, this can be an infinite loop
            in case of graph, unless, we store the visited cases! 
            The only way you can terminate is via return of BFS function
            that will interrupt the while loop. So, the function spits out the first solution it finds.
            '''
            ct += 1
            print(ct, end='\r')

            if len(frontier) > frontier_max_size:
                frontier_max_size = len(frontier)
            # This is a measure of memory requirements. Although storing explored elements kills
            # difference between DFS and BFS cause DFS is promoted as having small memory requirements, but when visited
            # nodes
            # are stored in addition to frontier, at some point, there will be no big difference.
            # This is a cost we pay to convert a graph to a tree.
            current_node = frontier.pop(0)  # select and remove the first node in the queue
            ft.remove(current_node.state.get_state_tuple())

            self.add_to_explored_states(current_node, explored)
            # The reason why it is stored as a tuple, is to make elements hashable, so then we can ask if an element is
            # in the list

            # Firstly, let's check if the new node is the goal:
            if current_node.goal_test():
                print('Time required = ', -start + time.time())
                print('Explored states = ', len(explored))
                print('Frontier max size = ', frontier_max_size)
                return self.done(current_node)

            # if it is not the goal, then, let's dig deeper:
            actions = self.explore_actions(current_node.state)  # branches that can come out
            for anaction in actions:  # add exploration results to the frontier.

                new_state = self.world_dynamics(current_node.state, anaction)

                if not self.state_exists_in_explorered_states(new_state, explored):
                    if not self.state_exists_in_frontier_states(new_state, ft):
                        new_node = NodeMod(state=new_state, parent=current_node, action=anaction,
                                        depth=current_node.depth + 1)
                        frontier.append(new_node)
                        ft.add(new_state.get_state_tuple())

        print('Failed to reach target goal. Number of states explored = ')
        return []  # i.e. frontier list was emptied, all state space was explored, goal wasn't reached.
        # The result returned above should be equal to half of the state space size since it was proven that the other
        # half is unsolvable. The state space cardinality is 9! half of them has parity "odd" and half got parity "even"
        # Meaning half of them fall into one subset and so is the other half. In each subset you can move freely
        # between any two states. Additionally, when starting in one subset, you're stuck in it, and thus, there are
        # only 9! / 2 states to explore.

# DFS node inherit BFS node
class Node1Mod(NodeMod):
    def __init__(self, state:SokobanMapMod, parent=None, action=None, depth=0):
        self.state = state  # a board
        self.parent = parent  # parent node, a NODE! not just a matrix.
        self.action = action  # The one that led to this node (useful for retracing purpose)
        self.depth = depth  # depth of the node in the tree. This is the criterion for who's next in DFS, BFS.

    def DFS(self):
        start = time.time()
        frontier = [self]  # queue of found but unvisited nodes, FILO
        ft = set()
        ft.add(self.state.get_state_tuple())
        # A version of the frontier that contains only states, not nodes
        # This is version is required because Python is slow, and having this numeric version
        # will make things much faster.
        frontier_max_size = len(frontier)
        explored = set([])
        ct = 0
        while frontier:
            ct += 1
            print(ct, end='\r')
            if len(frontier) > frontier_max_size: frontier_max_size = len(frontier)
            current_node = frontier.pop()  # FILO
            ft.remove(current_node.state.get_state_tuple())
            self.add_to_explored_states(current_node, explored)

            if current_node.goal_test():
                print('Time required = ', -start + time.time())
                print('Explored states = ', len(explored))
                print('Frontier max size = ', frontier_max_size)
                return self.done(current_node)
                return True  # This return is for BFS method. It is a mean to break out of the while loop.

            actions = self.explore_actions(current_node.state)  # branches that can come out
            for anaction in actions:  # add exploration results to the frontier.
                new_state = self.world_dynamics(current_node.state, anaction)
                if not self.state_exists_in_explorered_states(new_state, explored):
                    if not self.state_exists_in_frontier_states(new_state, ft):
                        new_node = Node1Mod(state=new_state, parent=current_node, action=anaction,
                                            depth=current_node.depth + 1)
                        frontier.append(new_node)
                        ft.add(new_state.get_state_tuple())
        print('Failed to reach target goal. Number of states explored = ')
        return []

# UCS Node inherit DFS and BFS nodes
class Node2Mod(Node1Mod):  # this way, we will inherit DFS and BFS methods
    def __init__(self, state:SokobanMapMod, path_cost=0, parent=None, action=None, depth=0):
        self.state = state  # a board
        self.parent = parent  # parent node, a NODE! not just a matrix.
        self.action = action  # The one that led to this node (useful for retracing purpose)
        self.depth = depth  # depth of the node in the tree. This is the criterion for who's next in DFS, BFS.
        self.path_cost = path_cost

    def whos_next_UCS(self, frontier):
        return np.argmin([anelement.path_cost for anelement in frontier])

    def UCS(self, astar_search=False):
        start = time.time()
        frontier = [self]
        frontier_max_size = len(frontier)
        explored = set([])

        while frontier:
            if len(frontier) > frontier_max_size: frontier_max_size = len(frontier)
            index = self.whos_next_UCS(frontier)
            current_node = frontier[index]  # select, then remove the first node in the queue
            del frontier[index]
            print(len(frontier), end='\r')

            self.add_to_explored_states(current_node, explored)

            if current_node.goal_test():
                print('Time required = ', -start + time.time())
                print('Explored states = ', len(explored))
                print('Frontier max size = ', frontier_max_size)
                return self.done(current_node)

            actions = self.explore_actions(current_node.state)  # branches that can come out
            for anaction in actions:  # add exploration results to the frontier.
                new_state = self.world_dynamics(current_node.state, anaction)
                if not self.state_exists_in_explorered_states(new_state, explored):
                    # Consider to add another check for child not being in frontier.
                    # Calculate heuristic cost
                    heuristic_cost = 0
                    if astar_search == True:
                        heuristic_cost = new_state.calculate_heuristic_cost()
                    new_node = Node2Mod(state=new_state, parent=current_node, action=anaction,
                                        depth=current_node.depth + 1,
                                        path_cost=current_node.path_cost + heuristic_cost)
                                        # path_cost=current_node.path_cost + 1)
                    frontier.append(new_node)
        print('Failed to reach target goal. Number of states explored = ')
        return []




# Main function
def main(arglist):
    try:
        getchar = msvcrt.getch
    except ImportError:
        getchar = sys.stdin.read(1)

    if len(arglist) == 0:
        print("Running this file directly launches a playable game of Sokoban based on the given map file.")
        print("Usage: sokoban_app.py [map_file_name] [output_file_name]")
        return

    f = open(arglist[0], 'r')

    # initial state
    map_init = SokobanMapMod(arglist[0])

    # root_node = NodeMod(map_init)
    # final_node = root_node.BFS()
    root_node = Node2Mod(map_init)
    solution = root_node.UCS(True)

    # write file
    if len(arglist) == 1:
        return
    step_count = 0
    with open(arglist[1], 'w') as output_file:
        for step in solution:
            step_count = step_count + 1
            output_file.write("%s" % step)
            if step_count < len(solution):
                output_file.write(",")



if __name__ == '__main__':
    main(sys.argv[1:])







