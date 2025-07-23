from controllers.controlTree import Controller
from random import random, choice, choices
import copy
import sys
from tqdm import tqdm

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

    tree1.updateDepth()
    tree2.updateDepth()

def mutate(tree): 
    # during mutation, pick a random parent node, and replace right/left with another subtree
    # make sure doesn't exceed max depth
    randParent = tree.selectRandom()
    
    if random() < 0.5: 
        subtree=Controller(maxDepth=tree.maxDepth - randParent.left.depth + 1)
        randParent.left = subtree.root
    else: 
        subtree=Controller(maxDepth=tree.maxDepth - randParent.right.depth + 1)
        randParent.right = subtree.root

    # update depth of nodes of trees
    tree.updateDepth()


# start with initial population of trees
# select parents of population
# crossover, mutate, or replicate to create offspring
# repeat

def naturalSelection(modelPath, dataPath, maxDepth, POP=100, GENERATIONS=1000):

    parents = [] 
    for _ in range(POP): 
        tree = Controller(maxDepth=maxDepth)
        tree.evalFitness(modelPath, dataPath)
        parents.append(tree)


    for i in tqdm(range(GENERATIONS)): 
        fit = [p.fitness for p in parents]
        totalFitness = sum(fit)
        print("Avg Fitness of Generation: ", totalFitness/len(parents))
        print("Max Fitness of Generation: ", max(fit))


        # # -100/-300 = 0.33, -200/-300 = 0.66
        # but -100 should actually be more likely
        # with negatives, 1 -
        probs = [1 - (p.fitness/totalFitness) for p in parents]

        # parents are selected like a roulette wheel (w/replacement)
        # e.g. there could be multiple copies of the same individual
        breeders = choices(parents, weights=probs, k=POP)

        # generate offspring by crossing over, replicating, or mutating
        offspring = [] 
        mated = set()
        for i, b in enumerate(breeders):
            # if already mated, skip
            if i in mated: 
                continue

            prob = random()

            if prob < 0.1: 
                offspring.append(b)
            elif prob < 0.3: 
                mutate(b)
                offspring.append(b)
            else: 
                if i < len(breeders) - 1: 
                    # crossover until both r valid
                    mateWith = choice(breeders[:i] + breeders[i+1:])
                    numGood = 0

                    # might get 1 extra offspring per crossover
                    while numGood < 2: 
                        crossover(b, mateWith)
                        # get the depth of tree
                        if(b.getDepth() <= b.maxDepth):
                            numGood += 1
                            offspring.append(b)
                        if(mateWith.getDepth() <= mateWith.maxDepth):
                            numGood += 1
                            offspring.append(mateWith)
                            mated.add(i)
            
        for o in offspring: 
            o.evalFitness(modelPath, dataPath)

        parents = offspring

    # in the end return the best control tree 
    bestFitness = -1 * sys.maxsize
    bestTree = None
    for p in parents: 
        if(p.fitness > bestFitness): 
            bestFitness = p.fitness 
            bestTree = p 

    return bestTree, bestFitness


if __name__ == "__main__":
    bestTree, bestFitness = naturalSelection(modelPath="models/tinyphysics.onnx", dataPath="data", maxDepth=5, POP=100, GENERATIONS=30)
    bestTree.printTree()
    print("Fitness: ", bestFitness)











