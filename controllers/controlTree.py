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
        self.functions = ["roll_lataccel", "v_ego", "a_ego"]
        self.constants = [1, 2, 3, 4, 5]
        self.terminals = self.functions + self.constants
        self.maxDepth=maxDepth

        # choose a random operator
        self.root = Node(choice(self.operators), depth=1)
        self.createTree(self.root)
        self.fitness = 0

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
        sim = TinyPhysicsSimulator(model, dataPath, self)

        return sim.rollout()

    def evalFitness(self, modelPath, dataPath, numRollouts=100): 
        files = sorted(Path(dataPath).iterdir())[:numRollouts]
        rolloutMap = partial(self.rollout, modelPath)
        
        # parallel speedup
        costs = process_map(rolloutMap, files)
        return sum(costs)/len(costs)

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
        if(root.depth + 1 == self.maxDepth): 
            # then bottom nodes must be terminals 
            root.right = Node(choice(self.terminals), self.maxDepth)
            root.left = Node(choice(self.terminals), self.maxDepth)

            return root

        # if it is a terminal, don't do anything (base case)
        if(root.me not in self.terminals): 
            # decide to assign a constant or a terminal to node
            right = Node(self.generate(), root.depth + 1)
            left = Node(self.generate(), root.depth + 1)

            root.right = self.createTree(right)
            root.left = self.createTree(left)

        return root
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        dicState = {"roll_lataccel": state.roll_lataccel, "v_ego": state.v_ego, "a_ego": state.a_ego}
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

'''
tree = ControlTree()
tree.selectRandom()
tree = ControlTree()
# tree.printTree()
state = {
    "root_lataccel": 3, 
    "v_ego": 2.1, 
    "a_ego": 1
}
print(tree.evaluate(state=state))
'''