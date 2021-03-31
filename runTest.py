"""
This is a test script which define a custom runGames function 
for our RL project use
"""

import random
from pacman import readCommand, ClassicGameRules
import RLAgents
from layout import getLayout
import numpy as np
import matplotlib.pyplot as plt
import sys
import textDisplay
import copy
import __main__
import time


def runGames(layoutName, pacman, ghosts, display, numGames, record, numTraining=0, catchExceptions=False, timeout=30, **args):
    """
    A copy of same function in pacman.py
    """
    __main__.__dict__['_display'] = display

    layout = getLayout(layoutName)
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
        ## save the result
        name =  getTitleName(pacman, layoutName, numGames)
        np.save(f"data/{name}.npy",{
            'wins' : wins,
            'scores': scores
        }, allow_pickle=True)

    return games

def getTitleName(pacman,layoutName,numGames, **args):
    """
    get the title name for each argument specification
    """
    pacmanName = type(pacman).__name__
    titleName = f"n={numGames}.{layoutName}.{pacmanName}"

    return titleName

def plotGames(args, show = False):
    """
    plot games
    """ 
    titleName = getTitleName(**args)
    dic = np.load(f"data/{titleName}.npy", allow_pickle=True).item()
    scores = dic['scores']
    wins = dic['wins']

    avgScoreList = []
    nGames = len(scores)
    # only look at the lastest window number of games 
    window = min(int(0.1*nGames),100)
    avgScoreList = [scores[0]]
    winRateList = []
    winCnt = 0
    for i in range(1,nGames):
        iend = min(i+window,nGames)
        scoreToLookAt = scores[i:iend]
        #scoreToLookAt = scores[0:i]
        winCnt += wins[i]
        #avgScoreList.append(np.mean(scoreToLookAt))
        avgScoreList.append(avgScoreList[-1]*(i-1)/i + scores[i]/i)
        winRateList.append(winCnt/(i+1))

    plt.figure(figsize=(14,6))
    plt.subplot(121)
    plt.plot(avgScoreList)
    plt.xlabel("number of games")
    plt.ylabel("average score")
    plt.subplot(122)
    plt.plot(winRateList)
    plt.xlabel("number of games")
    plt.ylabel("winning rate")

    titleName = getTitleName(**args)

    plt.savefig(f"figs/{titleName}.pdf",bbox_inches='tight')
    if show:
        plt.show()
        
def test(run=True):
    """
    The main test function by Junyan Su
    """
    # manually set the parameters here, please comment it out if you want to set them from command line
    #def runGames(layout, pacman, ghosts, display, numGames, record, numTraining=0, catchExceptions=False, timeout=30):
    random.seed('sdsc8006')
    argsOrigin = readCommand(sys.argv[1:])  # Get game components based on input
    argsOrigin['display'] = textDisplay.NullGraphics()

    layoutNames = ['mediumClassic', 'mediumGrid']
    pacmans = [
        RLAgents.MonteCarloAgent(eps0=1e1,gamma=1),
        RLAgents.QLearningAgent(eps0=1e1, gamma=1, alpha=1e-4),
    ]
    layoutNames = [layoutNames[1]]  # only choose one for testing
    pacmans = [pacmans[1]]
    
    argsList = []

    for pacman in pacmans:
        for layoutName in layoutNames:
            argsTmp = copy.deepcopy(argsOrigin)
            argsTmp['pacman'] = pacman
            argsTmp['layoutName'] = layoutName
            argsList.append(argsTmp)

    if run:
        for args in argsList:
            runGames(**args)
    
    for args in argsList:
        plotGames(args, show=False)

if __name__ == '__main__':
    """
    """
    np.seterr(all='raise')
    start = time.time()
    test(run=True)
    print(f"time used {time.time() - start:.2f} s")
