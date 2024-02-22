# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util
from functools import partial

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    
    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        "*** YOUR CODE HERE ***"
        minGhostDist = float('inf')
        for ghost in newGhostStates:
            distcurr = manhattanDistance(newPos, ghost.configuration.pos)
            if distcurr < minGhostDist:
                minGhostDist = distcurr
        
        food = currentGameState.getFood()
        #print(type(food))
        if food[newPos[0]][newPos[1]]:
            minFoodDist = 0
        else:
            minFoodDist = float('inf')
            for x in range(food.width):
                for y in range(food.height):
                    if food[x][y]:
                        distcurr = manhattanDistance(newPos, (x, y))
                        if distcurr < minFoodDist:
                            minFoodDist = distcurr
        gscore = 0
        fscore = 0
        if minGhostDist <= 0: # to avoid integer division by zero
            #minGhostDist +=1
            return -float('inf')
        else:
            gscore = 1 / (minGhostDist)
        # if minFoodDist <= 0:# to avoid integer division by zero
        #     #minFoodDist +=1
        #     fscore = 1 / (minFoodDist + 1)
        if minFoodDist >= 0:# to avoid integer division by zero
            #minFoodDist +=1
            fscore = 1 / (minFoodDist +1)
        # else:
        #     fscore = 1 / (minFoodDist)
            
        
        
        return fscore - gscore
    
def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        numGhosts = gameState.getNumAgents() - 1
        return self.maximize(gameState, 1, numGhosts)
    

    
    def maximize(self, gameState, depth, numGhosts): # we assume the first satte is max state
        if gameState.isWin() or gameState.isLose(): # We do the main work in minimize function as it can be the last leaf node ot it can lead to a max node
          return self.evaluationFunction(gameState)
        maxV = float("-inf")
        for action in gameState.getLegalActions(0): # for every current action available to pacman
          successor = gameState.generateSuccessor(0, action) #for every successor state for the determined aftert the action is taken
          '''if the value of the minimize(successor) is greater than maxV then:
                    maxV = minimize(successor)
            '''
          temp = self.minimize(successor, depth, 1, numGhosts)
          
          if temp > maxV:
                maxV = max(maxV, temp)
                best_action = action

        # terminal maxV returns actions whereas intermediate maxV returns values
        if depth > 1:
           return maxV
        else:
         return best_action
    
    def minimize(self, gameState, depth, agentIndex, numGhosts):
        if gameState.isWin() or gameState.isLose():
          return self.evaluationFunction(gameState)
        min = float("inf")
        actions = gameState.getLegalActions(agentIndex)
        successors = []
        for action in actions:# for every current action available to pacman
            successors.append(gameState.generateSuccessor(agentIndex, action))#for every successor state for the determined aftert the action is taken
        for successor in successors:
                if depth < self.depth and agentIndex == numGhosts: #maximize the utility of the next sucesor state that is pacman's turn
                    temp = self.maximize(successor, depth + 1, numGhosts)# we go one level deeper by calling (depth + 1)
                    if temp < min:
                        min = temp
                elif depth == self.depth and agentIndex == numGhosts: #leaf nodes
                    temp = self.evaluationFunction(successor) # on the leaf node there are no further nodes to explore and hence calling a minimize or maximize function would throw an error
                    if temp < min:# hence we call the evaluation function
                        min = temp
                elif agentIndex != numGhosts: # on the same level minimize the othe ghost's utility 
                    temp = self.minimize(successor, depth, agentIndex + 1, numGhosts)
                    if temp < min:
                        min = temp
        return min  

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        numGhosts = gameState.getNumAgents() - 1
        alpha = float("-inf") #stores maximum value found so far
        beta = float("inf") #stores minimum value found so far
        return self.maximize(gameState, 1, numGhosts, alpha, beta)
        util.raiseNotDefined()
        
    def maximize(self, gameState, depth, numGhosts, alpha, beta):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        maxV = float("-inf")
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            temp = self.minimize(successor, depth, 1, numGhosts, alpha, beta)
            if temp > maxV:
                maxV = temp
                best_action = action
            if maxV > beta:
                return maxV  # Pruning remaining branches
            alpha = max(alpha, maxV)
        if depth > 1:
            return maxV
        else:
            return best_action

    def minimize(self, gameState, depth, agentIndex, numGhosts, alpha, beta):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        minV = float("inf")
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            if depth == self.depth and agentIndex == numGhosts: #leaf node and hence doesnt maximise or minimize next successor but uses self.evaluationFunction
                    temp = self.evaluationFunction(successor)
            elif depth != self.depth and agentIndex == numGhosts: #maximizes the utility of the next sucesor state that is pacman's turn
                    temp = self.maximize(successor, depth + 1, numGhosts, alpha, beta) # goes deeper by calling (depth + 1)
            elif agentIndex != numGhosts: # minimizes the other ghost's utility
                    temp = self.minimize(successor, depth, agentIndex + 1, numGhosts, alpha, beta)# on the same level of tree it minimize the other ghost's utility
            if temp < minV:
                minV = temp
            if minV < alpha:
                return minV  # Pruning remaining branches
            beta = min(beta, minV)
        return minV
    
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    An expectimax agent for playing a game
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action from the current gameState using self.depth
        and self.evaluationFunction.
        """
        # Start from the pacman agent (index 0) and depth 0
        action = self.expectimax(gameState, 0, 0)
        #print(action[0] , "action[0]")
        #print(action[1] , "action1]")
        return action[0]

    def expectimax(self, gameState, agentIndex, depth):
        # If the game is over or the depth limit has been reached
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return None, self.evaluationFunction(gameState)

        numAgents = gameState.getNumAgents()
        nextAgent = (agentIndex + 1) % numAgents
        #print(nextAgent ,  " nextAgent" , "  Depth " , depth)
        #print("numAgents " , numAgents)
        if nextAgent == 0:
            nextDepth = depth + 1
        else:
            nextDepth = depth

        if agentIndex == 0:  # Maximizing for Pacman
            return self.maxValue(gameState, nextAgent, nextDepth)
        else:  # Expectation for Ghosts
            return self.expValue(gameState, nextAgent, nextDepth)

    def maxValue(self, gameState, nextAgent, depth): #for pacman
        maxScore = float("-inf")
        #maxAction = None
        for action in gameState.getLegalActions(0):  # Pacman's actions
            successor = gameState.generateSuccessor(0, action) #for every successor state 
            score = self.expectimax(successor, nextAgent, depth)
            #print(score[1] , " Score")
            if score[1] > maxScore: # getting the max score
                maxScore = score[1]
                maxAction = action
        #print(maxAction , "maxAction" , " maxScore " , maxScore)
        return maxAction, maxScore

    def expValue(self, gameState, nextAgent, depth): #for ghost
        totalScore = 0 
        actions = gameState.getLegalActions(nextAgent - 1)
        if len(actions) == 0:
            return None, self.evaluationFunction(gameState)
        for action in actions: # ghost's actions
            successor = gameState.generateSuccessor(nextAgent - 1, action)
            score = self.expectimax(successor, nextAgent, depth)
            totalScore += score[1]
        avgScore = totalScore / len(actions)
        return None, avgScore  

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = []
    for ghostState in newGhostStates:
        newScaredTimes.append(ghostState.scaredTimer)
    # Calculating the score from the current game state
    score = currentGameState.getScore()
    
    # Penalizeing states with more remaining food
    score -= len(newFood.asList())
    
    # Calculating distance to the nearest food
    foodDistances = []
    for foodPos in newFood.asList():
        foodDistances.append(manhattanDistance(newPos, foodPos))
    if foodDistances:
        score += 1.0 / min(foodDistances)
    
    # Adjusting score based on ghost distances and scared times
    for i in range(len(newGhostStates)):
        ghost = newGhostStates[i]
        scaredTime = newScaredTimes[i]
        ghostDistance = manhattanDistance(newPos, ghost.getPosition())
    
        if scaredTime > 0:
        # Encouraging chasing scared ghosts
            score += 2.0 / (ghostDistance + 1)
        else:
        # Avoiding close ghosts
            if ghostDistance <= 1:
                score -= 5


    return score

    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
