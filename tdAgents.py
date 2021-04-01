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

    def convertState(self, gameState):
        """
        The original hash function of gameState including scores which is useless in our MDP setting. 
        So in this function, we convert the gameState to a tuple of data we really care.
        """
        # list is not hashable, so we convert them to tuples
        state = (
            gameState.getPacmanState(),
            tuple(gameState.getGhostStates()),
            tuple(gameState.getCapsules()),
            gameState.getFood())
        #state = gameState
        return state

    def checkAction(self, gameState, action):
        if not (action in gameState.getLegalActions()):
            raise Exception("action not legal")
            

class TDAgent(RLAgent):
    """
    Use TD Control to play the pacman game
    Refer to the lesson slide 6: Model free control 
    Try the GLIE version
    """

    def __init__(self, numTraining = 0, eps0=1, gamma = 0.9999):
        super().__init__(numTraining=numTraining) # call initialize function of the parent class
        # The initial  \epsilon greedy exploration, 
        # for training index n, we set eps = initEps/n
        self.eps0 = float(eps0) 
        self.eps = self.eps0
        self.gamma = float(gamma)
        # the policy pi, which is a dict from state to an action
        self.pi = {}
        # the action-value function, which is a dict from (state,action) to a score
        self.Q = {}
        # the list that store one instance of episode, updated in the function self.final
        self.episode = []
        self.trainIndex = 0 #current traning index
        self.initState = True
        self.n=0
        self.alpha=1e-1
    
    def getQvalue(self, gameState, action):
        """
        get Q value 
        """
        self.checkAction(gameState,action)
        state = self.convertState(gameState)
        key = (state,action)
        if key in self.Q:
            return self.Q[key]
        else: # if it is not stored yet
            self.Q[key] = 0
            return 0

    def getAction(self, gameState):
        """
        get the action 
        """
        #initial state with random action
        if self.initState:
            self.initState = False
            legalMoves = gameState.getLegalActions()
            randInd = random.randint(0,len(legalMoves)-1)
            action = legalMoves[randInd]
            self.episode.append((gameState,action))
        else:
            legalMoves = gameState.getLegalActions()
            randNumber = random.uniform(0,1)
            if randNumber < self.eps:
                 # with probability eps, we use uniformly random actions for greedy exploration
                randInd = random.randint(0,len(legalMoves)-1)
                action = legalMoves[randInd]
            else:
                QList = [self.getQvalue(gameState, act) for act in legalMoves]
                    # if the Q value are all zeros, just randomly select one
                maxq = max(QList)
            # there may exist multiple max Q values, we simply randomly select them
                maxInds = [i for (i,q) in enumerate(QList) if q == maxq]
                randInd = random.randint(0,len(maxInds)-1)
                ind = maxInds[randInd]

            # find the index of the max Q value in the Q list
                action = legalMoves[ind]
            
            self.episode.append((gameState,action))
            (gameState0,FormerAction) = self.episode[-2]
            key = (self.convertState(gameState0),FormerAction)
            R = gameState.getScore() - gameState0.getScore()
            self.Q[key] = self.getQvalue(gameState0,FormerAction)+self.alpha*(R+self.gamma*self.getQvalue(gameState,action)-self.getQvalue(gameState0,FormerAction))
        self.n += 1
        self.eps = self.eps/self.n
        #print(legalMoves)
        return action

    def final(self, gameState):
        """
        final has been reached, we collect an episode now.
        let us update the policy accordingly
        """
        self.episode.append((gameState,))
        self.trainIndex += 1
        self.episode = [] # reset the episode to empty
        self.initState=True
      

    
        

