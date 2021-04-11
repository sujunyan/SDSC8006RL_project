"""
This is a test script which define a custom runGames function 
for our RL project use
"""

import random
from pacman import ClassicGameRules, default, loadAgent, parseAgentArgs
import layout
from layout import getLayout
import RLAgents
from tdAgents import TDAgent
import numpy as np
import matplotlib.pyplot as plt
import sys,os
import textDisplay
import copy
import __main__
import time
import PIL
from PIL import EpsImagePlugin


def readCommand(argv):
    """
    Processes the command used to run pacman from the command line.
    """
    from optparse import OptionParser
    usageStr = """
    USAGE:      python pacman.py <options>
    EXAMPLES:   (1) python pacman.py
                    - starts an interactive game
                (2) python pacman.py --layout smallClassic --zoom 2
                OR  python pacman.py -l smallClassic -z 2
                    - starts an interactive game on a smaller board, zoomed in
    """
    parser = OptionParser(usageStr)

    parser.add_option('-n', '--numGames', dest='numGames', type='int', default=10,
                      help=default('the number of GAMES to play'), metavar='GAMES')
    parser.add_option('--numDisplay', dest='numDisplay', type='int', default=3,
                      help=default('the number of GAMES to display'))
    parser.add_option('-l', '--layout', dest='layout', default = 'small',
                      help=default('the index of LAYOUT_FILE from which to load the map layout'))
    parser.add_option('-p', '--pacman', dest='pacman', default='MC',
                      help=default('the agent in the pacmanAgents module to use'))
    parser.add_option('-a', '--all', dest='testAll', action='store_true', default=False,
                      help=default('if we want to test all the parameters'))
    parser.add_option('-s', '--showPlot', dest='showPlot', action='store_true', default=False,
                      help=default('If we want to show the plot of the result'))
    parser.add_option('--norun', dest='noRun', action='store_true', default=False,
                      help=default('If run the game'))

    parser.add_option('--savegif', dest='savegif', action='store_true', default=False,
                      help=default('If save the gif files'))

    options, otherjunk = parser.parse_args(argv)

    if len(otherjunk) != 0:
        raise Exception('Command line input not understood: ' + str(otherjunk))

    args = dict()
    


    ghostType = loadAgent('RandomGhost', False)
    args['ghosts'] = [ghostType(i+1) for i in range(10)]
    args['numGames'] = options.numGames
    args['numGamesToDisplay'] = options.numDisplay
    argsList = []
    if options.testAll:
        for layout in layouts:
            for pacman in pacmans.values():
                argsTmp = copy.deepcopy(args)
                argsTmp['layoutName'] = layout
                argsTmp['pacman'] = pacman
                argsList.append(argsTmp)
    else:
        args['layoutName'] = options.layout
        args['pacman'] = pacmans[options.pacman]
        argsList.append(args)

    return (argsList, options)

def runGames(layoutName, pacman, ghosts, numGames, numGamesToDisplay = 1 ,numTraining=0, catchExceptions=False, timeout=30, **args):
    """
    A copy of same function in pacman.py
    """
    #__main__.__dict__['_display'] = display

    layout = getLayout(layoutName)
    rules = ClassicGameRules(timeout)
    games = []

    avgScore = 0
    winCnt = 0
    numGames = max(numGames,numGamesToDisplay)
    numTraining = numGames - numGamesToDisplay
    name =  getTitleName(pacman, layoutName, numGames)
    frameDir = f"gif/{name}"


    import shutil
    # delete older dir
    if os.path.exists(frameDir):
        shutil.rmtree(frameDir)
    os.mkdir(frameDir)

    for i in range(numGames):
        print(f"({i}/{numGames}) game start, avgScore {avgScore:.2f} winCnt {winCnt}")
        #beQuiet = i < numTraining
        beQuiet = (i < numTraining)
        if beQuiet:
            # Suppress output and graphics
            gameDisplay = textDisplay.NullGraphics()
            rules.quiet = True
        else:
            import graphicsDisplay
            gameDisplay = graphicsDisplay.PacmanGraphicsGif(
                zoom=1.0, capture=False, frameTime=0.01, storeFrameDir=frameDir, gameIdx=i-numTraining+1, totalGame=numGamesToDisplay)
            rules.quiet = False

        game = rules.newGame(layout, pacman, ghosts,
                             gameDisplay, beQuiet, catchExceptions)
        game.run()

        avgScore = (avgScore * i +  game.state.getScore())/(i+1)
        winCnt += game.state.isWin()
        games.append(game)

        #if not beQuiet:
        #    newFrames = game.display.frames
        #    nFrameToPause = 5
        #    frames.extend([newFrames[0] for i in range(nFrameToPause)])
        #    frames.extend(newFrames)
        #    frames.extend([newFrames[-1] for i in range(nFrameToPause)])

    # end of simulation of games
    # report and save the results
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

    np.save(f"data/{name}.npy",{
        'wins' : wins,
        'scores': scores
    }, allow_pickle=True)
    #frames = [PIL.Image.fromarray(f) for f in frames]
    #if frames:
    #    gifImg = frames[0]
    #    gifImg.save(f"gif/{name}.gif", format="GIF", append_images=frames, save_all=True, duration=100, loop=0)

    return games

def getTitleName(pacman, layoutName, numGames, **args):
    """
    get the title name for each argument specification
    """
    pacmanName = type(pacman).__name__
    titleName = f"n={numGames}.{layoutName}.{pacmanName}"

    return titleName

def getPlotData(args):
    """
    get the plot data
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
    return (avgScoreList, winRateList)

def plotAll(layoutNames, pacmans, numGames, show = False):
    """
    plot all games in argsList in a single file
    """
    
    args = dict()
    args['numGames'] = numGames
    for layoutName in layoutNames:
        plt.figure(figsize=(14,6))
        plt.subplot(121)
        plt.xlabel("number of games")
        plt.ylabel("average score")
        plt.subplot(122)
        plt.xlabel("number of games")
        plt.ylabel("winning rate")
        for pacman in pacmans:
            args['pacman'] = pacman
            args['layoutName'] = layoutName
            avgScoreList, winRateList = getPlotData(args) 
            label = type(pacman).__name__
            plt.subplot(121)
            plt.plot(avgScoreList, label=label)
            plt.subplot(122)
            plt.plot(winRateList, label=label)

        for opt in [121, 122]:
            plt.subplot(opt)
            plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=2, fancybox=True, shadow=True)

        plt.savefig(f"figs/total.n={args['numGames']}.{layoutName}.pdf",bbox_inches='tight')
    
    if show:
        plt.show()
        

def plotGames(args, show = False):
    """
    plot games
    """ 
    avgScoreList, winRateList = getPlotData(args) 

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
        
def getGif(args):
    """
    convert the frames to a gif file
    """
    from os import listdir
    from os.path import isfile, join
    name = getTitleName(**args)
    frameDir = f"gif/{name}"
    files = [f for f in listdir(frameDir) if isfile(join(frameDir, f))]
    gameTotal = 0
    imgDict = dict()
    frames = []
    for i,fName in enumerate(files):
        print(f"In getGif, loading {i}/{len(files)} files")
        fName_s = fName.split(".")
        gameIdx = int(fName_s[1])
        frameIdx = int(fName_s[2])
        gameTotal = max(gameTotal,gameIdx)
        imgTmp = PIL.Image.open(f"{frameDir}/{fName}")
        imgTmp.load()
        imgDict[(gameIdx,frameIdx)] = imgTmp
    
    
    nFrameToPause = 5
    keys = imgDict.keys()
    for iGame in range(1,gameTotal+1):
        imgTmp = imgDict[(iGame,1)]
        # at the begining, pause a little for user experience
        frames.extend([imgTmp for i in range(nFrameToPause)])
        iFrame = 0
        while True:
            #print(f"adding frame ({iGame},{iFrame})")
            iFrame +=1
            key = (iGame,iFrame)
            # this reach the end of the current game
            if key not in keys:
                # append last frames to pause
                frames.extend([imgTmp for i in range(nFrameToPause)])
                break
            imgTmp = imgDict[key]
            frames.append(imgTmp)

    gifImg = frames[0]
    gifImg.save(f"gif/{name}.gif", format="GIF", append_images=frames, save_all=True, duration=100, loop=0)

if __name__ == '__main__':
    """
    """
    start = time.time()

    EpsImagePlugin.gs_windows_binary =  r'C:\Program Files\gs\gs9.54.0\bin\gswin64c'
    np.seterr(all='raise')
    random.seed('sdsc8006')

    layouts = ['small', 'medium']
    pacmans = {
        'MC' : RLAgents.MonteCarloAgent(eps0=1e1,gamma=1),
        'TD' : TDAgent(eps0=1e1,gamma=1),
        #'QL' : RLAgents.QLearningAgent(eps0=1, gamma=1, alpha=1e-4),
        # alpha for w update, beta for theta update
        #'AC'   : RLAgents.ActorCriticAgent(gamma=1, alpha=1e-4, beta=1e-4),
    }

    argsList, options = readCommand(sys.argv[1:])

    if not options.noRun:
        for args in argsList:
            print(args)
            runGames(**args)
    
    if options.testAll:
        plotAll(layouts, pacmans.values(), options.numGames, show=options.showPlot)
    
    for args in argsList:
        if options.savegif:
            getGif(args)
        plotGames(args, options.showPlot)

    print(f"time used {time.time() - start:.2f} s")
