# qlearningAgents.py
# ------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random, util, math


class QLearningAgent(ReinforcementAgent):
    """
    Q-Learning Agent
    
    Functions you should fill in:
      - getQValue
      - getAction
      - getValue
      - getPolicy
      - update
      
    Instance variables you have access to
      - self.epsilon (exploration prob)
      - self.alpha (learning rate)
      - self.gamma (discount rate)
    
    Functions you should use
      - self.getLegalActions(state) 
        which returns legal actions
        for a state
  """

    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        "*** YOUR CODE HERE ***"
        self.Q_values = util.Counter()  #  {[state, action]: Q, [state, action]: Q}

    def getQValue(self, state, action):
        """
      Returns Q(state,action)    
      Should return 0.0 if we never seen
      a state or (state,action) tuple 
    """
        "*** YOUR CODE HERE ***"
        # return : Q (si,ai)
        if (state, action) not in self.Q_values.keys():
            return 0
        else:
            return self.Q_values[(state, action)]

    def getValue(self, state):
        """
      Returns max_action Q(state,action)        
      where the max is over legal actions.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return a value of 0.0.
    """
        "*** YOUR CODE HERE ***"
        # Q[state, best_action]
        values = [self.getQValue(state, action) for action in self.getLegalActions(state)]
        if values:
            return max(values)
        else:
            return 0.0


    def getPolicy(self, state):
        """
      Compute the best action to take in a state.  Note that if there
      are no legal actions, which is the case at the terminal state,
      you should return None.
    """
        "*** YOUR CODE HERE ***"
        legalActions = self.getLegalActions(state)

        #  no actions
        if len(legalActions) == 0:
            return None

        #  has actions
        q_value_list = []
        for action in legalActions:
            q_value = self.getQValue(state, action)
            q_value_list.append(q_value)

        best_action_index = q_value_list.index(max(q_value_list))
        return legalActions[best_action_index]

    def getAction(self, state):
        """
      Compute the action to take in the current state.  With
      probability self.epsilon, we should take a random action and
      take the best policy action otherwise.  Note that if there are
      no legal actions, which is the case at the terminal state, you
      should choose None as the action.
    
      HINT: You might want to use util.flipCoin(prob)
      HINT: To pick randomly from a list, use random.choice(list)
    """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None

        "*** YOUR CODE HERE ***"
        if util.flipCoin(self.epsilon):
            # take random action
            action = random.choice(legalActions)
        else:
            # take best policy
            action = self.getPolicy(state)
        return action

    def update(self, state, action, nextState, reward):
        """
      The parent class calls this to observe a 
      state = action => nextState and reward transition.
      You should do your Q-Value update here
      
      NOTE: You should never call this function,
      it will be called on your behalf
    """
        "*** YOUR CODE HERE ***"
        update_value = self.getQValue(state, action) + self.alpha*(reward + self.gamma * self.getValue(nextState) - self.getQValue(state, action))
        self.Q_values[(state, action)] = update_value

class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
        """
    These default parameters can be changed from the pacman.py command line.
    For example, to change the exploration rate, try:
        python pacman.py -p PacmanQLearningAgent -a epsilon=0.1
    
    alpha    - learning rate
    epsilon  - exploration rate
    gamma    - discount factor
    numTraining - number of training episodes, i.e. no learning after these many episodes
    """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
    Simply calls the getAction method of QLearningAgent and then
    informs parent of action for Pacman.  Do not change or remove this
    method.
    """
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
     ApproximateQLearningAgent

     You should only have to overwrite getQValue
     and update.  All other QLearningAgent functions
     should work as is.
  """

    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)

        # You might want to initialize weights here.
        "*** YOUR CODE HERE ***"
        # key is (s, a), value is a dictionary for each feature of (s,a)
        self.weight_dict = util.Counter()

    def getQValue(self, state, action):
        """
      Should return Q(state,action) = w * featureVector
      where * is the dotProduct operator
    """
        "*** YOUR CODE HERE ***"
        q_value = 0
        featureVector = self.featExtractor.getFeatures(state, action)

        for k, v in featureVector.items():
            if k in self.weight_dict.keys():
                q_value = q_value + self.weight_dict[k]*v
        return q_value

        # features = self.featExtractor.getFeatures(state,action)
        # QValue = 0.0
        #
        # for feature in features:
        #     QValue += self.weight_dict[feature] * features[feature]
        #
        # return QValue

    def update(self, state, action, nextState, reward):
        """
       Should update your weights based on transition
    """
        "*** YOUR CODE HERE ***"
        # 𝑤i = 𝑤i + + 𝛼[𝑐𝑜𝑟𝑟𝑒𝑐𝑡𝑖𝑜𝑛]𝑓i(𝑠, 𝑎)
        # 𝑐𝑜𝑟𝑟𝑒𝑐𝑡𝑖𝑜𝑛 =:𝑅(𝑠, 𝑎) + 𝛾𝑉(𝑠_prime) − 𝑄(𝑠, 𝑎)
        correction = reward + self.gamma * self.getValue(nextState) - self.getQValue(state, action)

        featureVector = self.featExtractor.getFeatures(state, action)
        for k, v in featureVector.items():
            if k in self.weight_dict.keys():
                self.weight_dict[k] = self.weight_dict[k] + self.alpha*correction*v
            else:
                self.weight_dict[k] = self.alpha*correction*v

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
# class ApproximateQAgent(PacmanQAgent):
#     """
#        ApproximateQLearningAgent
#        You should only have to overwrite getQValue
#        and update.  All other QLearningAgent functions
#        should work as is.
#     """
#     def __init__(self, extractor='IdentityExtractor', **args):
#         self.featExtractor = util.lookup(extractor, globals())()
#         PacmanQAgent.__init__(self, **args)
#         self.weights = util.Counter()
#
#     def getWeights(self):
#         return self.weights
#
#     def getQValue(self, state, action):
#         """
#           Should return Q(state,action) = w * featureVector
#           where * is the dotProduct operator
#         """
#
#         features = self.featExtractor.getFeatures(state,action)
#         QValue = 0.0
#
#         for feature in features:
#             QValue += self.weights[feature] * features[feature]
#
#         return QValue
#
#     def update(self, state, action, nextState, reward):
#         """
#            Should update your weights based on transition
#         """
#         QValue = 0
#         difference = reward + (self.alpha * self.getValue(nextState) - self.getQValue(state, action))
#         features = self.featExtractor.getFeatures(state, action)
#
#         for feature in features:
#           self.weights[feature] += self.alpha * features[feature] * difference
#
#
#     def final(self, state):
#         "Called at the end of each game."
#         # call the super-class final method
#         PacmanQAgent.final(self, state)
#
#         # did we finish training?
#         if self.episodesSoFar == self.numTraining:
#             # you might want to print your weights here for debugging
#             "*** YOUR CODE HERE ***"
#             pass