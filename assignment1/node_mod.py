import numpy as np
import time
from sokoban_map_mod import SokobanMapMod

class NodeMod():
    def __init__(self, state:SokobanMapMod, parent=None, action=None, depth=0):
        self.state = state  # a board
        self.parent = parent  # parent node, a NODE! not just a matrix.
        self.action = action  # The one that led to this node (useful for retracing purpose)
        self.depth = depth  # depth of the node in the tree. This is the criterion for who's next in DFS, BFS.

    def world_dynamics(self, current_state, action):
        if action == SokobanMapMod.UP:
            new_state = current_state.apply_move(SokobanMapMod.UP)
            if new_state is None:
                raise Exception('Inappropriate action for the current state')
        elif action == SokobanMapMod.DOWN:
            new_state =  current_state.apply_move(SokobanMapMod.DOWN)
            if new_state is None:
                raise Exception('Inappropriate action for the current state')
        elif action == SokobanMapMod.LEFT:
            new_state = current_state.apply_move(SokobanMapMod.LEFT)
            if new_state is None:
                raise Exception('Inappropriate action for the current state')
        elif action == SokobanMapMod.RIGHT:
            new_state = current_state.apply_move(SokobanMapMod.RIGHT)
            if new_state is None:
                raise Exception('Inappropriate action for the current state')
        else:
            print('Unknown action!')
        return new_state

    def explore_world_dynamics(self, current_state, action):
        if action == SokobanMapMod.UP:
            new_state = current_state.explore_apply_move(SokobanMapMod.UP)
            if new_state == False:
                raise Exception('Inappropriate action for the current state')
        elif action == SokobanMapMod.DOWN:
            new_state =  current_state.explore_apply_move(SokobanMapMod.DOWN)
            if new_state == False:
                raise Exception('Inappropriate action for the current state')
        elif action == SokobanMapMod.LEFT:
            new_state = current_state.explore_apply_move(SokobanMapMod.LEFT)
            if new_state == False:
                raise Exception('Inappropriate action for the current state')
        elif action == SokobanMapMod.RIGHT:
            new_state = current_state.explore_apply_move(SokobanMapMod.RIGHT)
            if new_state == False:
                raise Exception('Inappropriate action for the current state')
        else:
            print('Unknown action!')
        return new_state


    def explore_actions(self, state):
        possibilities = [SokobanMapMod.LEFT, SokobanMapMod.RIGHT, SokobanMapMod.UP, SokobanMapMod.DOWN]
        actions = []
        for apossibility in possibilities:
            try:
                new_state = self.explore_world_dynamics(state, apossibility)
                actions.append(apossibility)  # if world_dynamics didn't return False
            except:
                pass  # move on to the next possibility
        return actions

    def whos_next_BFS(self, frontier):
        """
        This is an ideal function. It relies on .depth attributes of nodes.
        And determines which node should be explored first according to BFS regime.
        However, there's a faster implementation of it. Which is FIFO frontier.
        This list by construction will make the most recent nodes (which are the deepest) be explored later
        While older nodes in the list go first. So, no need for this function. It is only here for perfection.
        Pythonically speaking, use .pop(0) and .append methods of lists to build a FIFO type queue
        """
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
                print(i, '\n')

    def BFS(self, goal_state):
        start = time.time()
        frontier = [self]  # queue of found but unvisited nodes, FIFO
        frontier_max_size = len(frontier)  # We measure space / memory requirements of the algo via this metric.
        ft = []
        ft.append(self.state)
        # A version of the frontier that contains only states, not nodes
        # This is version is required because Python is slow, and having this numeric version
        # will make things much faster.

        explored = []
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

            if len(frontier) > frontier_max_size: frontier_max_size = len(frontier)
            # This is a measure of memory requirements. Although storing explored elements kills
            # difference between DFS and BFS cause DFS is promoted as having small memory requirements, but when visited
            # nodes
            # are stored in addition to frontier, at some point, there will be no big difference.
            # This is a cost we pay to convert a graph to a tree.
            current_node = frontier.pop(0)  # select and remove the first node in the queue
            ft.remove(current_node.state)
            explored.append(current_node.state)  # cause we are going to explore it.
            # The reason why it is stored as a tuple, is to make elements hashable, so then we can ask if an element is
            # in the list

            # Firstly, let's check if the new node is the goal:
            if current_node.state == goal_state:
                print('Time required = ', -start + time.time())
                print('Explored states = ', len(explored))
                print('Frontier max size = ', frontier_max_size)
                self.done(current_node)
                '''
                Time required shows overall performance. This could be different indicator from explored states figure
                Because there can be a costy h function that leads to less explored states but more computation time.
                So, there is no point of it. Additionally, run time can depend on computer specs, and representation
                of states (a matrix or a string)
                Frontier max size is indication of storage requirements.
                Number of steps to reach the solution is indication of how optimal the solution  is.
                '''
                return True  # This return is for BFS method. It is a mean to break out of the while loop.

            # if it is not the goal, then, let's dig deeper:
            actions = self.explore_actions(current_node.state)  # branches that can come out
            for anaction in actions:  # add exploration results to the frontier.
                new_state = self.world_dynamics(current_node.state, anaction)
                if  new_state not in explored:
                    """ cause we are imposing a tree search on a graph problem
                    The problem has cycles, something that doesn't exist in tree searches. So, we have to
                    prevent that.
                    A tree is like exploring your folders in HDD. you cannot open a subfolder and it takes you
                    to an earlier folder. In graphs this is possible. An example of a graph is a map.
                    """
                    if new_state not in ft:
                        new_node = NodeMod(state=new_state, parent=current_node, action=anaction,
                                        depth=current_node.depth + 1)
                        frontier.append(new_node)
                        ft.add(new_state)

        print('Failed to reach target goal. Number of states explored = ')
        return len(explored)  # i.e. frontier list was emptied, all state space was explored, goal wasn't reached.
        # The result returned above should be equal to half of the state space size since it was proven that the other
        # half is unsolvable. The state space cardinality is 9! half of them has parity "odd" and half got parity "even"
        # Meaning half of them fall into one subset and so is the other half. In each subset you can move freely
        # between any two states. Additionally, when starting in one subset, you're stuck in it, and thus, there are
        # only 9! / 2 states to explore.