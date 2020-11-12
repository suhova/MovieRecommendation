import json
import math

import numpy as np
import pandas as pd
import requests
from SPARQLWrapper import SPARQLWrapper, JSON

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
        if (sqrtUser * sqrtPredictUser) == 0:
            simUsers = 0
        else:
            simUsers = sumuv / (sqrtUser * sqrtPredictUser)
        userSlice = [v for v in list(filter(lambda rating: rating != unseen, user[:-1]))]
        if len(userSlice) == 0:
            mean = 0
        else:
            mean = np.mean(userSlice)
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


def getMovieId(movieName):
    API_ENDPOINT = "https://www.wikidata.org/w/api.php"
    params = {
        'action': 'wbsearchentities',
        'format': 'json',
        'language': 'en',
        'search': movieName
    }
    res = requests.get(API_ENDPOINT, params=params)
    res.json()['search'][0]['description']
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql_query = """ SELECT $movie
          WHERE {
              ?movie wdt:P31 wd:Q11424;
                     wdt:P1476 ?title filter regex(?title,\"""" + movieName + """\",'i').
          }
    """
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    return results['results']['bindings'][0]['movie']['value']


def sparqlSelect(movieName):
    API_ENDPOINT = "https://www.wikidata.org/w/api.php"
    params = {
        'action': 'wbsearchentities',
        'format': 'json',
        'language': 'en',
        'search': movieName
    }
    res = requests.get(API_ENDPOINT, params=params)
    res.json()['search'][0]['description']
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")

    # movieId = getMovieId(movieName)
    movieId = "http://www.wikidata.org/entity/Q25188"
    movieId = movieId[movieId.rfind('/', 0, len(movieId)) + 1:]
    sparql_query = """ 
    SELECT ?actor ?actorLabel
WHERE{
  {SELECT ?actor (MIN(?pubdate) as ?minPubdate)
          WHERE {
            ?movie wdt:P161 ?actor filter(?movie = wd:""" + movieId + """).
            ?film wdt:P161 ?actor;
                  wdt:P577 ?pubdate.
          }
   GROUP BY ?actor
}
  ?firstFilm wdt:P577 ?date filter(?firstFilm = wd:""" + movieId + """)
             filter(?date = ?minPubdate).
  SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
}
    """
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    return results['results']['bindings']


data = pd.read_csv('hw21/data.csv', delimiter=', ', engine='python', index_col=0)
contextDay = pd.read_csv('hw21/context_day.csv', delimiter=', ', engine='python', index_col=0)
contextPlace = pd.read_csv('hw21/context_place.csv', delimiter=', ', engine='python', index_col=0)
movieNames = pd.read_csv('hw21/movie_names.csv', delimiter=', ', engine='python', index_col=0)
variant = 33
knn = 4
userName = 'User ' + str(variant)

firstTaskResult = firstTask(data, userName, knn)
secondTaskResult = secondTask(data, contextDay, contextPlace, userName, 15)
writeJson(variant, firstTaskResult, secondTaskResult)

for film in secondTaskResult:
    actors = sparqlSelect(list(movieNames[movieNames.index == film])[0])
    for actor in actors:
        print('Link: ' + actor['actor']['value'] + ' Name: ' + actor['actorLabel']['value'])
