"""
In this file, we implement our agent with Reinforcement Learning technique
"""

from pacman import GameState
from game import Agent
from util import manhattanDistance
from game import Directions
import random, util

class RLAgent(Agent):
    """
    This class provides some common elements to all of our Reinforcement Learning agents.
    Any methods defined here will be available for its child class.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, numTraining = 0):
        self.index = 0 # Pacman is always agent index 0
        self.numTraining = numTraining
        #self.evaluationFunction = util.lookup(evalFn, globals())
        #self.depth = int(depth)
    def convertState(self, gameState):
        """
        The original hash function of gameState including scores which is useless in our MDP setting. 
        So in this function, we convert the gameState to a tuple of data we really care.
        """
        state = (
            gameState.getPacmanPosition(),
            gameState.getGhostPositions(),
            gameState.getCapsules(),
            gameState.getFood())
        return state



class MonteCarloAgent(RLAgent):
    """
    Use Monte Carlo Control to play the pacman game
    Refer to the lesson slide 6: Model free control (11-19)
    """

    def __init__(self, numTraining = 0, eps=1e-2):
        super().__init__(numTraining=numTraining) # call initialize function of the parent class
        # the pol
        self.eps = eps # The parameter used in \epsilon greedy exploration
        # the policy pi, which is a dict from state to an action
        self.pi = {}
        # the action-value function , which is a dict from (MDPState) to a (action,score)
        self.Q = {}

    def getAction(self, gameState):
        """
        """

        legalMoves = gameState.getLegalActions()
        #print(legalMoves)
        return legalMoves[0]
        #print(self.convertState(gameState))

    def final(self, gameState):
        """
        final has been reached, we collect an episode now,
        let us update the policy accordingly
        """
        print("final reached")

    
        

