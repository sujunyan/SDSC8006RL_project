"""
From this part, we start to implement our agent with Reinforcement Learning technique
"""

class RLAgent(Agent):
    """
    This class provides some common elements to all of our Reinforcement Learning agents.
    Any methods defined here will be available for its child class.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        #self.evaluationFunction = util.lookup(evalFn, globals())
        #self.depth = int(depth)


class MonteCarloAgent(RLAgent):
    """
    Use Monte Carlo Control to play the pacman game
    """

    def __init__(self):
        super().__init__() # call initialize function of the parent class

    def getAction(self, gameState):
        """
        """
        util.raiseNotDefined()

