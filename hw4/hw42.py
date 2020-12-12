# Вариант 27
# 1.1 случайная генерация
# 2.2 выбрать только 20% самых приспособленных особей
# 3.2 однородный (каждый бит от случайно выбранного родителя)
# 4.3 добавление 1 случайной вещи 5% особей
# 5.3 замена своих родителей

import random
import numpy as np
import pandas as pd

data = pd.read_csv('27.txt', delimiter=' ', engine='python')
maxValues = data.columns.tolist()
maxW = int(maxValues[0])
maxV = int(maxValues[1])
data.columns = ['v', 'c']
data['w'] = data.index
countOfItems = len(data)
initialPopulation = 200
data.index = np.arange(countOfItems, dtype=object)
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


def isItemFitInBackpack(df):
    return sum(df["w"]) < maxW and sum(df["v"]) < maxV


def createInitialPopulation(sizeOfPopulation=initialPopulation):
    return [createIndividual() for _ in range(0, sizeOfPopulation)]


def createIndividual():
    rndList = [i for i in range(0, countOfItems) if random.randint(0, 1) == 1]
    rndIndividual = data[data.index.isin(rndList)]
    rndIndividual.index = np.arange(len(rndIndividual))
    while not isItemFitInBackpack(rndIndividual):
        rnd = random.randint(0, len(rndIndividual) - 1)
        rndIndividual = rndIndividual.drop(rnd)
        rndIndividual.index = np.arange(len(rndIndividual))
    return rndIndividual


def chooseBestItems(arr, percent=0.2):
    size = round(len(arr) * percent)
    arr.sort(key=lambda df: grade(df), reverse=True)
    return arr[0:size], arr[size:len(arr)]


def crossingOver(mother, father):
    maxSize = max(len(mother), len(father))
    rndList = [random.randint(0, 1) for _ in range(0, maxSize)]
    motherList = [i for i in range(0, len(rndList)) if rndList[i] == 0]
    fatherList = [i for i in range(0, len(rndList)) if rndList[i] == 1]
    item = pd.concat([mother[mother.index.isin(motherList)], father[father.index.isin(fatherList)]], ignore_index=True) \
        .drop_duplicates()
    item.index = np.arange(len(item))
    return item


def grade(df):
    if not isItemFitInBackpack(df):
        return 0
    else:
        return sum(df['c'])


def generationGrade(generation):
    res = 0
    for i in range(0, len(generation)):
        res += grade(generation[i])
    return res


def mutation(generation, populationSize=200, percent=0.05):
    size = round(populationSize * percent)
    random.shuffle(generation)
    for i in range(0, size):
        generation[i] = addRandomItem(generation[i])
    return generation


def addRandomItem(df):
    length = len(df)
    rnd = random.randint(0, countOfItems - 1)
    while len(df) == length:
        df = pd.concat([df, data[data.index == rnd]], ignore_index=True) \
            .drop_duplicates()
        df.index = np.arange(len(df), dtype=object)
        rnd += 1
        if (rnd == countOfItems):
            rnd = 0
    return df


def GA(maxNoResultIterations=5, iter=20):
    population = createInitialPopulation()
    oldGr = generationGrade(population)
    while maxNoResultIterations != 0 and iter != 0:
        best, notBest = chooseBestItems(population)
        newGeneration = list()
        while len(best) > 0:
            random.shuffle(best)
            mother = best.pop()
            father = best.pop()
            child1 = crossingOver(mother, father)
            child2 = crossingOver(mother, father)
            if isItemFitInBackpack(child1):
                newGeneration.append(child1)
            else:
                newGeneration.append(mother)
            if isItemFitInBackpack(child2):
                newGeneration.append(child2)
            else:
                newGeneration.append(father)
        z = list(np.insert(notBest, np.arange(len(newGeneration)), newGeneration))
        population = mutation(z)
        newGr = generationGrade(population)
        if 0.3 * initialPopulation > abs(oldGr - newGr):
            maxNoResultIterations -= 1
        iter -= 1
        oldGr = newGr
    res, _ = chooseBestItems(population)
    return res[0]


result = GA()
print(grade(result))
print(result)
result.to_csv('result42.csv', index=False)
