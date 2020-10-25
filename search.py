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
        util.raiseNotDefined()

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


def depthFirstSearch(problem):
    # we tried, but it doesn't work :( 

    state = problem.getStartState()
    visitedNodes = dict()
    action = []
    DFS(problem, state, action, visitedNodes)
    return action

def DFS(problem, state, action, visitedNodes):
    if not problem.visitedNodes.has_key(hash(state[0])):
        problem.visitedNodes[hash(state[0])] = state[0]
        
    if problem.isGoalState(state):
        return True

    for child in problem.getSuccessors(state):
        problem.expandedAccumulator += problem._expanded

        state = child[0]
        if not problem.visitedNodes.has_key(hash(state)):
            problem.visitedNodes[hash(state)] = state
            action.append(child[1])
            if DFS(problem, state, action, visitedNodes) == True:
                return True
            action.pop()
    return False


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    from util import Queue
    frontier = Queue()
    visited = []

    if problem.isGoalState(problem.getStartState()):
        return []

    frontier.push((problem.getStartState(), []))

    while not frontier.isEmpty():
        head, path = frontier.pop()
        if head not in visited:
            visited.append(head)
        if problem.isGoalState(head):
            return path
        for succ, action, cost in problem.getSuccessors(head):
            problem.expandedAccumulator += problem._expanded
            if succ not in visited:
                if succ not in (item[0] for item in frontier.list):
                    frontier.push((succ, path + [action]))
    
    util.raiseNotDefined()
def randomSearch(problem):
    import random

    state = problem.getStartState()
    sol = []

    while not problem.isGoalState(state):
        successors = problem.getSuccessors(state)
        problem.expandedAccumulator += problem._expanded

        successor = random.choice(successors)
        state = successor[0]
        sol.append(successor[1])

    return sol

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    def cost_function(node): return node[2]

    frontier = util.PriorityQueueWithFunction(cost_function)

    explored = set()

    # Node: (state, action, cost, parent)
    frontier.push((problem.getStartState(), None, 0, None))

    while not frontier.isEmpty():
        node = frontier.pop()
        state = node[0]
        cost = node[2]

        if state in explored:
            continue

        explored.add(state)

        if problem.isGoalState(state):
            path = []
            while node[3] is not None:
                path = path + [node[1]]
                node = node[3]
            path.reverse()
            return path

        for child_state, child_action, child_cost in problem.getSuccessors(state):
            problem.expandedAccumulator += problem._expanded
            if child_state not in explored:
                frontier.push((child_state, child_action, cost + child_cost, node))

    return False

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    visited = {}
    solution = []
    queue = util.PriorityQueue()
    parents = {}
    cost = {}

    start = problem.getStartState()
    queue.push((start, 'Undefined', 0), 0)
    visited[start] = 'Undefined'
    cost[start] = 0

    if problem.isGoalState(start):
        return solution

    goal = False;
    while(queue.isEmpty()!=True and goal!=True):
        node = queue.pop()
        visited[node[0]] = node[1]
        if problem.isGoalState(node[0]):
            node_sol = node[0]
            goal = True
            break
        for elem in problem.getSuccessors(node[0]):
            problem.expandedAccumulator += problem._expanded
            if elem[0] not in visited.keys():
                priority = node[2] + elem[2] + heuristic(elem[0], problem)
                if elem[0] in cost.keys():
                    if cost[elem[0]] <= priority:
                        continue
                queue.push((elem[0], elem[1], node[2] + elem[2]), priority)
                cost[elem[0]] = priority
                parents[elem[0]] = node[0]

    while(node_sol in parents.keys()):
        node_sol_prev = parents[node_sol]
        solution.insert(0, visited[node_sol])
        node_sol = node_sol_prev
        
    return solution

    

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
