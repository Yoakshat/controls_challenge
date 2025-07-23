from random import random, choice
from tinyphysics import TinyPhysicsSimulator, TinyPhysicsModel
from . import BaseController
from functools import partial
from tqdm.contrib.concurrent import process_map
from pathlib import Path


# features: p, i, d, difference in rolling lat accell, a, (future diff in accel)
class Controller(): 
    def __init__(self, maxDepth=10): 
        self.operators = ["+", "x", "/", "-"]
        # our terminals
        self.functions = ["P", "I", "D"]
        self.constants = [1, 2, 3, 4, 5]
        self.terminals = self.functions + self.constants
        self.maxDepth=maxDepth

        self.errorIntegral = 0
        self.prevError = 0

        # choose a random operator
        self.root = Node(choice(self.operators), depth=1)

        if maxDepth == 1: 
            self.root = Node(choice(self.terminals), depth=1)
        else: 
            self.createTree(self.root)

        self.fitness = 0

    def getDepth(self): 
        # get the depth of the tree (go until leaf nodes)
        return self.getDepthRec(self.root)
    
    def getDepthRec(self, node):     
        if node.right is None: 
            return node.depth

        rightDepth = self.getDepthRec(node.right)
        leftDepth = self.getDepthRec(node.left)

        return max(rightDepth, leftDepth)
    
    # update depth of entire tree
    def updateDepth(self): 
        return self.updateDepthRec(self.root, 1)
    
    def updateDepthRec(self, node, depth): 
        if node is not None: 
            node.depth = depth

            self.updateDepthRec(node.right, depth+1)
            self.updateDepthRec(node.left, depth+1)    

    # select a random parent node
    def selectRandom(self): 
        self.parentNodes = [] 
        self.addRec(self.parentNodes, self.root)

        return choice(self.parentNodes)

    def addRec(self, allNodes, node):
        # add parent nodes (so if a node doesn't have a right node)
        if node.right is not None: 
            allNodes.append(node)

            self.addRec(allNodes, node.right)
            self.addRec(allNodes, node.left)

    def rollout(self, modelPath, dataPath): 
        model = TinyPhysicsModel(modelPath, debug=False)
        sim = TinyPhysicsSimulator(model, str(dataPath), self)

        return sim.rollout()

    def evalFitness(self, modelPath, dataPath, numRollouts=10): 
        files = sorted(Path(dataPath).iterdir())[:numRollouts]
        rolloutMap = partial(self.rollout, modelPath)
        
        # parallel speedup
        costs = process_map(rolloutMap, files)
        costs = [c['total_cost'] for c in costs]
        
        # lower the cost, the higher the fitness
        # cost is 100, fitness is -100
        self.fitness = -1 * sum(costs)/len(costs)

    # for testing
    def printTree(self):
        print(self.printTreeRec(self.root))

    def printTreeRec(self, node):
        
        if node is None:
            return ""
        
        right_str = self.printTreeRec(node.right)
        left_str = self.printTreeRec(node.left)
        
        return f"({right_str} {node.me} {left_str})"

    def generate(self): 
        if random() < 0.5: 
            # choose operator
            return choice(self.operators)
        else: 
            # pick terminal
            return choice(self.terminals)

    def createTree(self, root): 
        # if it is a terminal, don't do anything (base case)
        if(root.me not in self.terminals): 
            if(root.depth + 1 == self.maxDepth): 
                # then bottom nodes must be terminals 
                root.right = Node(choice(self.terminals), self.maxDepth)
                root.left = Node(choice(self.terminals), self.maxDepth)
            else: 
                # decide to assign a constant or a terminal to node with 50% probability each
                right = Node(self.generate(), root.depth + 1)
                left = Node(self.generate(), root.depth + 1)

                root.right = self.createTree(right)
                root.left = self.createTree(left)

        return root
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        error = (target_lataccel - current_lataccel)
        self.errorIntegral += error
        errorDiff = error - self.prevError
        self.prevError = error

        dicState = {"P": error, "I": self.errorIntegral, "D": errorDiff}
        return self.evaluate(dicState)

    # here state is a dictionary
    def evaluate(self, state): 
        return self.evaluateRec(self.root, state)
    
    def evaluateRec(self, root, state): 
        if root.me in self.operators: 
            rightRes = self.evaluateRec(root.right, state)
            leftRes = self.evaluateRec(root.left, state)

            if root.me == "+": 
                return rightRes + leftRes
            if root.me == "-": 
                return rightRes - leftRes
            if root.me == "x": 
                return rightRes * leftRes
            if root.me == "/": 
                if leftRes == 0: 
                    leftRes = 1
                return rightRes / leftRes
        else: 
            # it is a terminal node
            if root.me in self.constants: 
                return root.me
            else: 
                return state[root.me]
    
class Node(): 
    def __init__(self, me, depth): 
        self.left = None
        self.right = None
        self.me = me
        self.depth = depth

