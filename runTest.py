"""
This is a test script which define a custom runGames function 
for our RL project use
"""

import random
from pacman import readCommand, ClassicGameRules
import RLAgents
import layout
import numpy as np
import matplotlib.pyplot as plt
import sys
import textDisplay
import __main__

def runGames(layout, pacman, ghosts, display, numGames, record, numTraining=0, catchExceptions=False, timeout=30):
    """
    A copy of same function in pacman.py
    """
    __main__.__dict__['_display'] = display

    rules = ClassicGameRules(timeout)
    games = []


    for i in range(numGames):
        print(f"({i}/{numGames}) game start")
        #beQuiet = i < numTraining
        beQuiet = True
        if beQuiet:
            # Suppress output and graphics
            gameDisplay = textDisplay.NullGraphics()
            rules.quiet = True
        else:
            gameDisplay = display
            rules.quiet = False
        game = rules.newGame(layout, pacman, ghosts,
                             gameDisplay, beQuiet, catchExceptions)
        game.run()
        #if not beQuiet:
        #    games.append(game)

        games.append(game)
        if record:
            import time
            import pickle
            fname = ('recorded-game-%d' % (i + 1)) + \
                '-'.join([str(t) for t in time.localtime()[1:6]])
            f = file(fname, 'w')
            components = {'layout': layout, 'actions': game.moveHistory}
            pickle.dump(components, f)
            f.close()

    if (numGames-numTraining) > 0:
        scores = [game.state.getScore() for game in games]
        wins = [game.state.isWin() for game in games]
        winRate = wins.count(True) / float(len(wins))
        print('Average Score:', sum(scores) / float(len(scores)))
        #print('Scores:       ', ', '.join([str(score) for score in scores]))
        print('Win Rate:      %d/%d (%.2f)' %
              (wins.count(True), len(wins), winRate))
        #print('Record:       ', ', '.join(
        #    [['Loss', 'Win'][int(w)] for w in wins]))

    return games

def plotGames(games):
    """
    plot games
    """ 
    scores = [game.state.getScore() for game in games]
    wins = [game.state.isWin() for game in games]

    avgScoreList = []
    nGames = len(games)
    # only look at the lastest window number of games 
    window = int(0.1*nGames)
    avgScoreList = [scores[0]]
    winRateList = []
    winCnt = 0
    for i in range(1,nGames):
        #iend = min(i+window,nGames)
        #scoreToLookAt = scores[i:iend]
        scoreToLookAt = scores[0:i]
        winCnt += wins[i]
        #avgScoreList.append(np.mean(scoreToLookAt))
        avgScoreList.append(avgScoreList[-1]*(i-1)/i + scores[i]/i)
        winRateList.append(winCnt/(i+1))
    plt.subplot(121)
    plt.plot(avgScoreList)
    plt.xlabel("number of games")
    plt.ylabel("average score")
    plt.subplot(122)
    plt.plot(winRateList)
    plt.xlabel("number of games")
    plt.ylabel("winning rate")
    plt.show()
        
def testMCAgent():
    """
    set up the parameters to test the MonteCarloAgent 
    """
    args = readCommand(sys.argv[1:])  # Get game components based on input
    # manually set the parameters here, please comment it out if you want to set them from command line
    #def runGames(layout, pacman, ghosts, display, numGames, record, numTraining=0, catchExceptions=False, timeout=30):
    random.seed('sdsc8006')
    # eps0 = 10 seems a good choice, this means we need more random samples at the beginning
    args['pacman'] = RLAgents.MonteCarloAgent(eps0=1e1,gamma=0.9999)
    # the simplest layout
    args['layout'] = layout.getLayout('testClassic')
    # sufficient to see the point start to win
    args['numGames']  = 200
    args['display'] = textDisplay.NullGraphics()
    games = runGames(**args)
    plotGames(games)



if __name__ == '__main__':
    """
    """
    testMCAgent()
