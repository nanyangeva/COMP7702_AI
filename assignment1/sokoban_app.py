import sys
import msvcrt
import numpy as np
from sokoban_map_mod import SokobanMapMod
from node_mod import NodeMod



def main(arglist):
    try:
        getchar = msvcrt.getch
    except ImportError:
        getchar = sys.stdin.read(1)

    if len(arglist) != 1:
        print("Running this file directly launches a playable game of Sokoban based on the given map file.")
        print("Usage: sokoban_map.py [map_file_name]")
        return

    f = open(arglist[0], 'r')

    rows = []
    for line in f:
        if len(line.strip()) > 0:
            rows.append(list(line.strip()))

    f.close()

    row_len = len(rows[0])
    for row in rows:
        assert len(row) == row_len, "Mismatch in row length"

    num_rows = len(rows)
    num_boxes = 0
    box_positions = []
    tgt_positions = []
    player_position = None
    for i in range(num_rows):
        for j in range(row_len):
            if rows[i][j] == SokobanMapMod.BOX_SYMBOL:
                box_positions.append((i, j))
                rows[i][j] = SokobanMapMod.FREE_SPACE_SYMBOL
                num_boxes = num_boxes + 1
            elif rows[i][j] == SokobanMapMod.TGT_SYMBOL:
                tgt_positions.append((i, j))
                rows[i][j] = SokobanMapMod.FREE_SPACE_SYMBOL
            elif rows[i][j] == SokobanMapMod.PLAYER_SYMBOL:
                player_position = (i, j)
                rows[i][j] = SokobanMapMod.FREE_SPACE_SYMBOL
            elif rows[i][j] == SokobanMapMod.BOX_ON_TGT_SYMBOL:
                box_positions.append((i, j))
                tgt_positions.append((i, j))
                rows[i][j] = SokobanMapMod.FREE_SPACE_SYMBOL
            elif rows[i][j] == SokobanMapMod.PLAYER_ON_TGT_SYMBOL:
                player_position = (i, j)
                tgt_positions.append((i, j))
                rows[i][j] = SokobanMapMod.FREE_SPACE_SYMBOL

    assert len(box_positions) == len(tgt_positions), "Number of boxes does not match number of targets"

    # initial state
    x_size = row_len
    y_size = num_rows
    box_positions = box_positions
    tgt_positions = tgt_positions
    player_position = player_position
    player_x = player_position[1]
    player_y = player_position[0]
    obstacle_map = rows

    tgt_distances = []

    occupied_tgt_indexes = []

    # for each box, set other boxes as obstacles
    initial_box_positions = []
    goal_box_positions = []
    tgt_position_index = 0
    obstacle_map = rows
    # loop each box for moving purpose, each loop treat one box and return a intermediate result
    for i in range(num_boxes):
        # refresh initial box positions and goal box positions
        initial_box_positions.clear()
        goal_box_positions.clear()
        # only the current box is treated as a box, others will be treated as obstacles
        initial_box_positions.append(box_positions[i])

        # measure the L1 distance from the current box to all target locations
        for j in range(num_boxes):
            l1_distance = 99999999999;
            # check whether the target position has been occupied already, if yes, set distance to infinite
            if j in occupied_tgt_indexes:
                l1_distance = 99999999999
            else:
                l1_distance = np.abs(box_positions[i][0] - tgt_positions[j][0]) + np.abs(box_positions[i][1] - tgt_positions[j][1])
            tgt_distances.append(l1_distance)
        # minimum L1 distance is chosen to be the current box's goal position
        l1_distance_min = min(tgt_distances)
        tgt_position_index = 0
        for j in range(num_boxes):
            if j in occupied_tgt_indexes:
                continue
            else:
                if tgt_distances[j] == l1_distance_min:
                    tgt_position_index = j
                    break
        goal_box_position = tgt_positions[tgt_position_index]
        goal_box_positions.append(goal_box_position)

        # create goal state for the current box being processed
        # other boxes are treated as obstacles
        # occupied target positions are treated as obstacles
        # for j in range(num_boxes):
            # if j <= i:
                # obstacle_map[other_box_position[0]][other_box_position[1]] = SokobanMapMod.OBSTACLE_SYMBOL
            # else:
                # get other boxes location and treat them as obstacles
                # other_box_position = box_positions[j]
                # obstacle_map[other_box_position[0]][other_box_position[1]] = SokobanMapMod.OBSTACLE_SYMBOL
                # continue

        # for occupied target positions, treat them as obstacles
        for k in occupied_tgt_indexes:
            obstacle_map[tgt_positions[k][0]][tgt_positions[k][1]] = SokobanMapMod.OBSTACLE_SYMBOL

        # solve the current box
        # initial state
        map_init = SokobanMapMod(arglist[0])
        map_init.set_map(initial_box_positions, tgt_positions, player_position, obstacle_map)

        # goal state
        map_goal = SokobanMapMod(arglist[0])
        map_goal.set_map(goal_box_positions, tgt_positions, player_position, obstacle_map)

        root_node = NodeMod(map_init)
        root_node.BFS(map_goal)

        # if goal state is reached,
        occupied_tgt_indexes.append(tgt_position_index)




if __name__ == '__main__':
    main(sys.argv[1:])







