"""
In this file, we implement our agent with Reinforcement Learning technique
"""

from pacman import GameState
from game import Agent
from util import manhattanDistance
from game import Directions
import random, util
import numpy as np

class RLAgent(Agent):
    """
    This class provides some common elements to all of our Reinforcement Learning agents.
    Any methods defined here will be available for its child class.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, numTraining = 0, eps0=1e-2, gamma= 0.9999, alpha= 1e-2):
        self.index = 0 # Pacman is always agent index 0
        self.numTraining = numTraining
        # The initial  \epsilon greedy exploration, 
        # for training index n, we set eps = eps0/n
        self.eps0 = float(eps0) 
        self.eps = self.eps0
        self.gamma = float(gamma)
        # the parameter to discount the Q value so that it can "forget" the old values quickly
        self.alpha = alpha
        # the policy pi, which is a dict from state to an action
        self.pi = {}
        # the action-value function, which is a dict from (state,action) to a score
        self.Q = {}
        self.episode = []

    def convertState(self, gameState):
        """
        The original hash function of gameState including scores which is useless in our MDP setting. 
        So in this function, we convert the gameState to a tuple of data we really care.
        """
        # list is not hashable, so we convert them to tuples
        
        pacmanState = gameState.getPacmanState()
        ghostStates = gameState.getGhostStates()
        ghostPositions = [tuple(g.getPosition()) for g in ghostStates]
        ghostScaredTimers = [g.scaredTimer for g in ghostStates]
        capsules = gameState.getCapsules()
        
        state = (
            pacmanState,
            tuple(ghostStates),
            tuple(capsules),
            gameState.getFood()
            )
        #state = gameState.deepCopy()
        return state

    def checkAction(self, gameState, action):
        if not (action in gameState.getLegalActions()):
            raise Exception("action not legal")
    

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
        return action
            

class MonteCarloAgent(RLAgent):
    """
    Use Monte Carlo Control to play the pacman game
    Refer to the lesson slide 6: Model free control (11-23)
    Try the GLIE version
    """

    def __init__(self, numTraining = 0, eps0=1e-2, gamma= 0.9999, alpha= 1e-2):
        # call initialize function of the parent class
        super(). __init__(numTraining = numTraining, eps0=eps0, gamma= gamma, alpha= alpha)
        
        # the counter for each state action pair
        self.N = {}
        # the list that store one instance of episode, updated in the function self.final
        self.trainIndex = 0 #current traning index

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

    def updateQ(self, episode):
        """
        called at the end of an episode
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
            oldQ = self.getQvalue(gameState,action)
            self.Q[key] = ( (N-1)*oldQ + GList[t]) / N
            #self.Q[key] = (1-self.alpha)* oldQ + self.alpha * GList[t]

    def final(self, gameState):
        """
        final has been reached, we collect an episode now.
        let us update the policy accordingly
        """
        self.episode.append((gameState,))
        self.updateQ(self.episode)
        self.episode = [] # reset the episode to empty
        self.trainIndex += 1
        self.eps = self.eps0/self.trainIndex

    
## Start to implement function approximated based methods, 
##

class FunctionApproxAgent(RLAgent):
    """
    This class provides some common elements to all of our Function Approximate based agents.

    Note: this is an abstract class: one that should not be instantiated.  It's only partially specified, and designed to be extended.

    See lec 7. Value Function Approximation
    The only difference for its child classes is how to update the weight delta w
    """
    def __init__(self, numTraining, eps0, gamma, alpha):
        super().__init__(numTraining=numTraining, eps0=eps0, gamma=gamma, alpha=alpha)
        self.w = None
        self.lastAction = None
        
    def featureMap(self, gameState, action):
        """
        In function approximate agents, we reduce the dimension of state
        by considering only the "features" :
        1. the pacman position
        2. the positions of the ghosts
        3. the hash value of positions of capsules
        4. the scareTimer
        5. hash value of the foods
        """

        feature = []
        pacmanState = gameState.getPacmanState()
        ghostStates = gameState.getGhostStates()

        pacmanPosition = pacmanState.getPosition()
        ghostPositions = [ g.getPosition() for g in ghostStates]
        ghostScaredTimers = [g.scaredTimer for g in ghostStates]
        capsules = gameState.getCapsules()

        for iList in [pacmanPosition, ghostScaredTimers]:
            for i in iList:
                feature.append(i)
        
        for pos in ghostPositions:
            for i in pos:
                feature.append(i)

        feature.append(hash(gameState.getFood()))
        feature.append(hash(tuple(capsules)))
        return np.array(feature)
    
    def getQvalue(self, gameState, action):
        feature = self.featureMap(gameState, action)
        # initialize the weight vector
        if self.w is None:
            self.wLen = len(feature)
            self.w = np.zeros(self.wLen)
        return self.w.dot(feature)
    
    def target(self, gameState):
        """
        the target used in the method updateQBeforeAction,
        refer to  page 28/30 in the lec 7 slide
        """
        util.raiseNotDefined()
        
    def updateWBeforeAction(self,gameState):
        """
        update the Q before the action
        should be called in Q-learning process
        """
        # ignore the first state
        if self.lastAction:
            feature = self.featureMap(self.lastState, self.lastAction)
            R = gameState.getScore() - self.lastState.getScore()
            lastQ = self.getQvalue(self.lastState, self.lastAction)
            target = self.target(gameState)
            deltaW = R + self.gamma * target - lastQ
            deltaW = deltaW * self.alpha * feature
            self.w += deltaW

            
    def getAction(self, gameState):
        """
        """
        self.updateWBeforeAction(gameState)
        action =  super().getAction(gameState)
        self.lastAction = action
        self.lastState = gameState
        return action

    def final(self, gameState):
        """
        final has been reached, we collect an episode now.
        let us update the policy accordingly
        """
        self.updateWBeforeAction(gameState)

class QLearningAgent(FunctionApproxAgent):
    def __init__(self, numTraining = 0, eps0 = 1, gamma = 0.9999, alpha = 1e-4):
        super().__init__(numTraining, eps0, gamma, alpha)
    
    def target(self, gameState):
        legalMoves = gameState.getLegalActions()
        if legalMoves:
            QList = [self.getQvalue(gameState,a) for a in legalMoves]
            return max(QList)
        else:
            # if we are at the final state
            return 0


