from controllers.controlTree import Controller
from random import random, choice, choices
import copy
from tinyphysics import TinyPhysicsModel

# cross two trees over 
# pick the parent node for each 
# and decide to swap l/r with l/r
def crossover(tree1, tree2): 
    parent1 = tree1.selectRandom()
    parent2 = tree2.selectRandom()

    # neat way to swap instead of manually testing all combos
    swaps = [
        ("right", "left"), 
        ("left", "right"), 
        ("left", "left"), 
        ("right", "right")
    ]

    swap1, swap2 = choice(swaps)

    temp = copy.deepcopy(getattr(parent1, swap1))
    setattr(parent1, swap1, copy.deepcopy(getattr(parent2, swap2)))
    setattr(parent2, swap2, temp)

def mutate(tree): 
    # during mutation, pick a random parent node, and replace right/left with another subtree
    randParent = tree.selectRandom()
    subtree = Controller()

    if random() < 0.5: 
        randParent.left = subtree.root
    else: 
        randParent.right = subtree.root


# start with initial population of trees
# select parents of population
# crossover, mutate, or replicate to create offspring
# repeat

def naturalSelection(modelPath, dataPath, POP=100, GENERATIONS=1000):

    parents = [] 
    for _ in range(POP): 
        tree = Controller(maxDepth=10)
        tree.evalFitness(modelPath, dataPath)
        parents.append(tree)


    for _ in range(GENERATIONS): 
        totalFitness = sum([p.fitness for p in parents])
        probs = [p.fitness/totalFitness for p in parents]

        # parents are selected like a roulette wheel (w/replacement)
        # e.g. there could be multiple copies of the same individual
        breeders = choices(parents, weights=probs, k=POP)

        # generate offspring by crossing over, replicating, or mutating
        offspring = [] 
        for i, b in enumerate(breeders):
            prob = random()

            if prob < 0.1: 
                offspring.append(b)
            elif prob < 0.3: 
                mutate(b)
                offspring.append(b)
            else: 
                if i < len(breeders) - 1: 
                    mateWith = choice(breeders[:i] + breeders[i+1:])
                    crossover(b, mateWith)
                    offspring.append(b)
                    offspring.append(mateWith)

                    breeders.remove(mateWith)
            
        for o in offspring: 
            o.evalFitness(modelPath, dataPath)

        parents = offspring

    # in the end return the best control tree 
    bestFitness = -100000
    bestTree = None
    for p in parents: 
        if(p.fitness > bestFitness): 
            bestFitness = p.fitness 
            bestTree = p 

    return bestTree


if __name__ == "__main__":
    bestTree = naturalSelection(modelPath="models/tinyphysics.onnx", dataPath="data", POP=100, GENERATIONS=3)
    bestTree.printTree()











