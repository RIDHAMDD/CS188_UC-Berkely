# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

# Ridham Dholaria 
import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        #util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    
    
    
    stack = util.Stack()
    stack.push((problem.getStartState(), []))
    
    
    visited = []
    
    while True:
        
        if stack.isEmpty():
            return []
        
        
        # remove from the stack
        node = stack.pop()
        #print("stack poped", node)
        #print("Node type - ", type(node))
        
        #print("problem.isGoalState(node[0]) - ",node[0],problem.isGoalState(node[0]))
                # add the node to the visited set
        visited.append(node[0])
        
        # if the node contains a goal state then return the corresponding solution
        if problem.isGoalState(node[0]):
            #print("node[1] - ",node[1], "node[0] - ", node[0])
            return node[1]
        
        #print the visited node list
        #print("visited node list - ", visited)
        
        # expand the current node and add the resulting nodes to the stack
        for subnodes in problem.getSuccessors(node[0]): #add all posible directions that can be visited to stack
            if subnodes[0] not in visited:
                stack.push((subnodes[0], node[1] + [subnodes[1]]))
                #print("node[1] + [subnodes[1]] - ",node[1] + [subnodes[1]])
               # print("node[1] - ",node[1])
               # print("[subnodes[1]] - ",subnodes[1])  
    
def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    # Initializing the queue with start state and empty path
    StartState = problem.getStartState()
    q = util.Queue()
    q.push(StartState)
    #richie Ogura's (TA) Idea to use dictionary
    # Initializing the dictionary which will store visited states and their paths
    visited = {}
    visited[StartState] = []

    #BFS algorithm
    while not q.isEmpty():
        # Poping the node
        CurrentState = q.pop()

        # Checking for goal state
        if problem.isGoalState(CurrentState):
            return visited[CurrentState]
        else:
            # Expanding the current state
            for SuccessorState, action, Cost in problem.getSuccessors(CurrentState):
                # Checking if the successor state is already visited
                if SuccessorState not in visited:
                    # Adding the successor state to the visited dictionary  and its path aswell
                    visited[SuccessorState] = visited[CurrentState] + [action]
                    # Addiong the successor state to the queue
                    q.push(SuccessorState)

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    StartState = problem.getStartState() # getting start state
    pq = util.PriorityQueue() #intializing priority queue
    pq.push((StartState, []), 0) #pushing first node
    visited = set() #initializing set

    while not pq.isEmpty(): #UCS
        CurrentState, CurrentPath = pq.pop()
        if problem.isGoalState(CurrentState): #checking if destination reached
            return CurrentPath
        elif CurrentState in visited: #checking if already visited
            continue
        else:
            visited.add(CurrentState) #mark as visited
            for SuccessorState, action, Cost in problem.getSuccessors(CurrentState): #for possible successors
                successor_cost = problem.getCostOfActions(CurrentPath + [action]) #getting cost of successor
                successor_node = (SuccessorState, CurrentPath + [action]) #getting successor node
                pq.push(successor_node, successor_cost) #pushing successor node to priority queue


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    StartState = problem.getStartState()
    #dictionary idea by Riche Ogura (TA)
    StartNode = {'0': StartState, '1': [], '2': 0}
    #0 is state, 1 is path, 2 is cost
    pq = util.PriorityQueue() #initializing priority queue
    pq.push(StartNode, heuristic(StartState, problem)) #pushing first node
    visited = set()#initializing set
    
    #A*
    while not pq.isEmpty():
        CurrentNode = pq.pop()#poping
        CurrentState = CurrentNode['0']
        
        
        if problem.isGoalState(CurrentState): # Checking if the destination is reached
            return CurrentNode['1'] #returning the path
        elif CurrentState in visited:#checking if already visited
            continue
        else:
            CurrentPath, CurrentCost = CurrentNode['1'], CurrentNode['2']
            visited.add(CurrentState) #mark as visited
            # Expanding the current state
            for SuccessorState, action, Cost in problem.getSuccessors(CurrentState):
                SuccessorNode = {'0': SuccessorState, '1': CurrentPath + [action], '2': CurrentCost + Cost}
                #making new successor node with aded cost and path
                pq.push(SuccessorNode, CurrentCost + Cost + heuristic(SuccessorState, problem)) #pushing in pq
    

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
