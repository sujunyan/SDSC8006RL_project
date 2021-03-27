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
            gameState.getFood())
        #state = gameState
        return state

    def checkAction(self, gameState, action):
        if not (action in gameState.getLegalActions()):
            raise Exception("action not legal")
            

class MonteCarloAgent(RLAgent):
    """
    Use Monte Carlo Control to play the pacman game
    Refer to the lesson slide 6: Model free control (11-23)
    Try the GLIE version
    """

    def __init__(self, numTraining = 0, eps0=1e-2, gamma = 0.9999):
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
        # the counter for each state action pair
        self.N = {}
        # the list that store one instance of episode, updated in the function self.final
        self.episode = []
        self.trainIndex = 0 #current traning index
    
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

    def incrementCounter(self, gameState, action):
        """
        increment the corresponding counter by one
        return: the incremented counter
        """
        self.checkAction(gameState,action)
        state = self.convertState(gameState)
        key = (state,action)
        if key in self.N:
            self.N[key] += 1
        else: # if it is not stored yet
            self.N[key] = 1
        return self.N[key]

    def getAction(self, gameState):
        """
        """
        legalMoves = gameState.getLegalActions()
        randNumber = random.uniform(0,1)
        randInd = random.randint(0,len(legalMoves)-1)
        if randNumber < self.eps:
            # with probability eps, we use uniformly random actions for greedy exploration
            action = legalMoves[randInd]
        else:
            QList = [self.getQvalue(gameState, act) for act in legalMoves]
            # if the Q value are all zeros, just randomly select one
            if all(q == 0 for q in QList):
                ind = randInd
            else:
                ind = QList.index(max(QList)) 

            # find the index of the max Q value in the Q list
            action = legalMoves[ind]
            #print(legalMoves)
        self.episode.append((gameState,action))
        return action

    def updateQ(self, episode):
        """
        """
        rewardList = []
        for i in range(len(episode)-1):
            gameState = episode[i][0]
            gameStateNext = episode[i+1][0]
            # The reward at the current time
            R = gameStateNext.getScore() - gameState.getScore()
            rewardList.append(R)

        # construct G_t for each t, we should do it backward
        # G_{t-1} = R_t + gamma*G_{t}
        GList = [rewardList[-1] ] 
        # Should be careful about the range
        for t in range(len(rewardList)-2,-1,-1):
            Rt = rewardList[t]
            Gt = Rt + self.gamma * GList[0]
            GList.insert(0,Gt)

        # update the Q function
        for t in range(len(GList)):
            (gameState,action) = episode[t]
            key = (self.convertState(gameState),action)
            N = self.incrementCounter(gameState,action)
            self.Q[key] = ( (N-1)*self.getQvalue(gameState,action) + GList[t]) / N
            #self.Q[key] = ( (N-1)*self.Q[key] + GList[t]) / N

    def final(self, gameState):
        """
        final has been reached, we collect an episode now,
        let us update the policy accordingly
        """
        self.episode.append((gameState,))
        self.updateQ(self.episode)
        self.episode = [] # reset the episode to empty
        self.trainIndex += 1
        self.eps = self.eps0/self.trainIndex

    
        

