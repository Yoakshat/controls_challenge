from random import randint, random

class ControlTree(): 
    def __init__(self): 
        self.operators = ["+", "x", "/", "-"]
        # our terminals
        self.functions = ["roll_lataccel", "v_ego", "a_ego"]
        self.constants = [1, 2, 3, 4, 5]
        self.terminals = self.functions + self.constants

        # choose a random operator
        self.root = Node(self.operators[randint(0, len(self.operators)-1)])
        self.createTree(self.root)

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
            return self.operators[randint(0, len(self.operators)-1)]
        else: 
            # pick terminal
            return self.terminals[randint(0, len(self.terminals)-1)]

    def createTree(self, root): 
        # if it is a terminal, don't do anything (base case)
        if(root.me not in self.terminals): 
            # decide to assign a constant or a terminal to node
            right = Node(self.generate())
            left = Node(self.generate())

            root.right = self.createTree(right)
            root.left = self.createTree(left)

        return root

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
                if root.me == "roll_lataccel": 
                    return state.roll_lataccel
                if root.me == "v_ego": 
                    return state.v_ego
                if root.me == "a_ego": 
                    return state.a_ego

    
class Node(): 
    def __init__(self, me): 
        self.left = None
        self.right = None
        self.me = me

class TestState(): 
    def __init__(self): 
        self.root_lataccel = 3
        self.v_ego = 2.1
        self.a_ego = 1

    
tree = ControlTree()
tree.printTree()
state = TestState()
print(tree.evaluate(state=state))