"""
This is a test script which define a custom runGames function 
for our RL project use
"""

from pacman import readCommand, ClassicGameRules
import numpy as np
import sys

def runGames(layout, pacman, ghosts, display, numGames, record, numTraining=0, catchExceptions=False, timeout=30):
    import __main__
    __main__.__dict__['_display'] = display

    rules = ClassicGameRules(timeout)
    games = []


    for i in range(numGames):
        print(f"({i}/{numGames}) game start")
        beQuiet = i < numTraining
        if beQuiet:
                # Suppress output and graphics
            import textDisplay
            gameDisplay = textDisplay.NullGraphics()
            rules.quiet = True
        else:
            gameDisplay = display
            rules.quiet = False
        game = rules.newGame(layout, pacman, ghosts,
                             gameDisplay, beQuiet, catchExceptions)
        game.run()
        if not beQuiet:
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

    import matplotlib.pyplot as plt
    scores = [game.state.getScore() for game in games]
    avgScoreList = []
    nGames = len(games)
    # only look at the lastest window number of games 
    window = int(0.1*nGames)
    for i in range(nGames):
        #iend = min(i+window,nGames)
        scoreToLookAt = scores[0:i]
        #scoreToLookAt = scores[i:iend]
        avgScoreList.append(np.mean(scoreToLookAt))
    plt.plot(avgScoreList)
    plt.xlabel("number of games")
    plt.ylabel("average score")
    plt.show()
        

if __name__ == '__main__':
    """
    The main function called when pacman.py is run
    from the command line:

    > python pacman.py

    See the usage string for more details.

    > python pacman.py --help
    """
    args = readCommand(sys.argv[1:])  # Get game components based on input
    games = runGames(**args)
    plotGames(games)

    # import cProfile
    # cProfile.run("runGames( **args )")
    pass
