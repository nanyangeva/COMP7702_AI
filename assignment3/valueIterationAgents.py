# valueIterationAgents.py
# -----------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):                             # ValueEstimateionAgent : learning Agents.py
  """
      * Please read learningAgents.py before reading this.*

      A ValueIterationAgent takes a Markov decision process
      (see mdp.py) on initialization and runs value iteration
      for a given number of iterations using the supplied
      discount factor.
  """
  def __init__(self, mdp, discount = 0.9, iterations = 100):
    """
      Your value iteration agent should take an mdp on
      construction, run the indicated number of iterations
      and then act according to the resulting policy.
    
      Some useful mdp methods you will use:
          mdp.getStates()
          mdp.getPossibleActions(state)
          mdp.getTransitionStatesAndProbs(state, action)
          mdp.getReward(state, action, nextState)
    """
    self.mdp = mdp                          # object : ValueEstimationAgent
    self.discount = discount
    self.iterations = iterations
    self.values = util.Counter()            # A Counter: {state:value, state:value} -> empty ? what is this values should like ?
    "*** YOUR CODE HERE ***"
    values_i = {}
    values_i_plus_1 = {}
    # initialize values_i where i = 0
    states = mdp.getStates()
    for state in states:
      values_i[state] = 0
    # iterations
    for i in range(0, self.iterations):
      for state, value in values_i.items():
        # all possible actions from the given state
        actions = self.mdp.getPossibleActions(state)

        value_all_actions = []
        for action in actions:
          successors = mdp.getTransitionStatesAndProbs(state, action)
          sum_s_prime = 0
          for successor in successors:
            state_prime = successor[0]
            probability = successor[1]
            value_s_prime = values_i[state_prime]
            reward = mdp.getReward(state, action, state_prime)
            sum_s_prime = sum_s_prime + probability*(reward + value_s_prime*self.discount)
          value_all_actions.append(sum_s_prime)

        values_i_plus_1[state] = 0
        if len(value_all_actions) > 0:
          values_i_plus_1[state] = max(value_all_actions)

      values_i = values_i_plus_1
      values_i_plus_1 = {}

    self.values = values_i


    self.Q_values = util.Counter()


  def getValue(self, state):
    """
      Return the value of the state (computed in __init__).
    """
    return self.values[state]

  def getQValue(self, state, action):
    """
      The q-value of the state action pair
      (after the indicated number of value iteration
      passes).  Note that value iteration does not
      necessarily create this quantity and you may have
      to derive it on the fly.
    """
    "*** YOUR CODE HERE ***"
    successors = self.mdp.getTransitionStatesAndProbs(state, action)
    sum_s_prime = 0
    for successor in successors:
      state_prime = successor[0]
      probability = successor[1]
      value_s_prime = self.getValue(state_prime)
      reward = self.mdp.getReward(state, action, state_prime)
      sum_s_prime = sum_s_prime + probability * (reward + value_s_prime*self.discount)
    return sum_s_prime

  def getPolicy(self, state):
    """
      The policy is the best action in the given state
      according to the values computed by value iteration.
      You may break ties any way you see fit.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return None.
    """
    "*** YOUR CODE HERE ***"
    actions = self.mdp.getPossibleActions(state)

    if len(actions) == 0:
      return None

    q_value_all_actions = []
    for action in actions:
      q_value = self.getQValue(state, action)
      q_value_all_actions.append(q_value)
    best_action_index = q_value_all_actions.index(max(q_value_all_actions))
    return actions[best_action_index]

  def getAction(self, state):
    "Returns the policy at the state (no exploration)."
    return self.getPolicy(state)

