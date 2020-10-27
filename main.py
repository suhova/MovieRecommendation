import json
import math

import numpy as np
import pandas as pd


def recommendation(dat, userDat, n=4, unseen=-1, minRating=1, accuracy=3):
    data = dat.copy()
    userData = userDat.copy()
    listUserData = userData.values[0]
    meanPredictUser = np.mean([rating for rating in list(filter(lambda rating: rating != unseen, listUserData))])
    sim = []
    data['user'] = data.index
    for user in data.values:
        viewedFilms = list(
            filter(lambda rating: rating[0] != unseen and rating[1] != unseen, zip(listUserData, user[:-1])))
        sumuv = np.sum([u * v for u, v in viewedFilms])
        sqrtUser = np.sqrt(np.sum([v ** 2 for u, v in viewedFilms]))
        sqrtPredictUser = np.sqrt(np.sum([u ** 2 for u, v in viewedFilms]))
        simUsers = sumuv / (sqrtUser * sqrtPredictUser)
        mean = np.mean([v for v in list(filter(lambda rating: rating != unseen, user[:-1]))])
        sim.append((user[-1], np.round(simUsers, accuracy), np.round(mean, accuracy)))
    sim = [num for num in sim if not (math.isnan(num[1]) | math.isnan(num[2]))]
    knn = pd.DataFrame(sim)
    knn = knn.sort_values(by=[1], ascending=False)[1:n + 1]
    knn = data.merge(knn, left_on='user', right_on=0, how='inner')
    knn.rename(columns={1: 'sim', 2: 'mean'}, inplace=True)
    movies = list(userData.loc[:, (userData == unseen).all(axis=0)])
    result = []
    for movie in movies:
        users = knn[knn[movie] != unseen]
        if len(users) != 0:
            sumsim = np.sum(np.abs(users['sim']))
            sum = np.sum(users['sim'] * (users[movie] - users['mean']))
            prediction = np.around(sum / sumsim + meanPredictUser, accuracy)
            if prediction < minRating:
                prediction = minRating
            result.append((
                str.lower(movie),
                prediction
            ))
    return result


def firstTask(data, userToPredict, knn):
    userData = data.loc[(data.index == userToPredict)]
    result = recommendation(data, userData, knn)
    returnedList = {}
    for prediction in result:
        returnedList.update({prediction[0]: prediction[1]})
    return returnedList


def secondTask(data, context_day, context_place, userToPredict, knn, unseen=-1, accuracy=3):
    allMovies = list(data)
    userData = data.loc[(data.index == userToPredict)]
    merged = context_day.merge(context_place, left_index=True, right_index=True, how='inner') \
        .merge(data, left_index=True, right_index=True, how='inner')
    for movie in allMovies:
        merged.loc[~merged[movie + "_x"].isin(["Sat", "Sun"]), movie] = unseen
        merged.loc[merged[movie + "_y"] != "h", movie] = unseen
    result = pd.DataFrame(recommendation(merged[allMovies], userData, knn))
    avgRating = np.around(
        np.mean([rating for rating in list(filter(lambda rating: rating != unseen, userData.values[0]))]), accuracy)
    result = result[(result[1] >= avgRating)].values
    returnedList = {}
    for movie in result:
        returnedList.update({movie[0]: movie[1]})
    return returnedList


def writeJson(userNum, firstTaskResult, secondTaskResult, fileName='result.json'):
    result = {"user": userNum,
              "1": firstTaskResult,
              "2": secondTaskResult
              }
    with open(fileName, 'w') as f:
        json.dump(result, f, indent=10)


data = pd.read_csv('data.csv', delimiter=', ', index_col=0)
contextDay = pd.read_csv('context_day.csv', delimiter=', ', index_col=0)
contextPlace = pd.read_csv('context_place.csv', delimiter=', ', index_col=0)
variant = 33
knn = 4
userName = 'User ' + str(variant)
writeJson(variant, firstTask(data, userName, knn), secondTask(data, contextDay, contextPlace, userName, 15))
