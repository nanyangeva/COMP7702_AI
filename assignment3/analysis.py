# analysis.py
# -----------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

######################
# ANALYSIS QUESTIONS #
######################

# Change these default values to obtain the specified policies through
# value iteration.

def question2():
  # answerDiscount = 0.9
  # answerNoise = 0.2
  # Q2 solved, do not change it
  answerDiscount = 0.9
  answerNoise = 0
  return answerDiscount, answerNoise

def question3a():
  # refer the close exit (+1), risking the cliff (-10)
  # 3a solved: do not change it
  answerDiscount = 0.2
  answerNoise = 0.0
  answerLivingReward = 0.1
  return answerDiscount, answerNoise, answerLivingReward
  # If not possible, return 'NOT POSSIBLE'

def question3b():
  # Prefer the close exit (+1), but avoiding the cliff (-10)
  # 3b solved: do not change it
  answerDiscount = 0.2
  answerNoise = 0.1
  answerLivingReward = 0.0
  return answerDiscount, answerNoise, answerLivingReward
  # If not possible, return 'NOT POSSIBLE'

def question3c():
  # Prefer the distance exit (+10), risking the cliff (-10)
  # 3c solved: do not change it
  answerDiscount = 0.5
  answerNoise = 0.0
  answerLivingReward = 0.9
  return answerDiscount, answerNoise, answerLivingReward
  # If not possible, return 'NOT POSSIBLE'

def question3d():
  # Prefer the distance exit (+10), but avoiding the cliff (-10)
  # 3d solved: do not change it
  answerDiscount = 0.9
  answerNoise = 0.2
  answerLivingReward = 0.0
  return answerDiscount, answerNoise, answerLivingReward
  # If not possible, return 'NOT POSSIBLE'

def question3e():
  # avoid both exits
  # 3e solved: do not change it
  answerDiscount = 0.1
  answerNoise = 0.7
  answerLivingReward = 0.5
  return answerDiscount, answerNoise, answerLivingReward
  # If not possible, return 'NOT POSSIBLE'

def question6():
  # answerEpsilon = None
  # answerLearningRate = None
  # return answerEpsilon, answerLearningRate
  return 'NOT POSSIBLE'
  # If not possible, return 'NOT POSSIBLE'
  
if __name__ == '__main__':
  print('Answers to analysis questions:')
  import analysis
  for q in [q for q in dir(analysis) if q.startswith('question')]:
    response = getattr(analysis, q)()
    print('  Question %s:\t%s' % (q, str(response)))
