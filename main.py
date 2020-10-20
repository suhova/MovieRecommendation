import json
import math

import numpy as np
import pandas as pd


def recomendation(dat, dat33, n=4):
    data = dat.copy()
    data33 = dat33.copy()
    user33 = data33.values[0]
    ru = np.around(np.mean([rating for rating in list(filter(lambda rating: rating != -1, user33))]), 3)
    sim = []
    data['user'] = data.index

    for user in data.values:
        viewedFilms = list(filter(lambda rating: rating[0] != -1 and rating[1] != -1, zip(user33, user[:-1])))
        sumuv = np.sum([u * v for u, v in viewedFilms])
        sqrtv = np.sqrt(np.sum([v ** 2 for u, v in viewedFilms]))
        sqrtu = np.sqrt(np.sum([u ** 2 for u, v in viewedFilms]))
        simuv = np.around(sumuv / (sqrtv * sqrtu), 3)
        rv = np.around(np.mean([v for u, v in viewedFilms]), 3)
        sim.append((user[-1], simuv, rv))
    sim = [num for num in sim if not (math.isnan(num[1]) | math.isnan(num[2]))]
    sim.sort(key=lambda x: x[1], reverse=True)
    sim = sim[1:n+1]
    knn = pd.DataFrame(sim)
    knn = data.merge(knn, left_on='user', right_on=0, how='inner')
    knn.rename(columns={1: 'sim', 2: 'rv'}, inplace=True)
    movies = list(data33.loc[:, (data33 == -1).all(axis=0)])
    result = []
    for movie in movies:
        users = knn[knn[movie] != -1]
        if len(users) != 0:
            sumsim = np.sum(np.abs(users['sim']))
            sum = np.sum(users['sim'] * (users[movie] - users['rv']))
            prediction = np.around(sum / sumsim + ru, 3)
            if prediction < 1:
                prediction = 1
            result.append((
                str.lower(movie),
                prediction
            ))
    return result


# varianat 33
data = pd.read_csv('data.csv', index_col=0)
context_day = pd.read_csv('context_day.csv', index_col=0)
context_place = pd.read_csv('context_place.csv', index_col=0)

# Task 1
data33 = data.loc[(data.index == 'User 33')]
res1 = recomendation(data, data33)
result1 = {}
for movie in res1:
    result1.update({movie[0]: movie[1]})

# Task 2
# Алгоритм такой же, но только среди фильмов, просмотренных в выходные дома
allMovies = list(data)
merged = context_day.merge(context_place, left_index=True, right_index=True, how='inner') \
    .merge(data, left_index=True, right_index=True, how='inner')
for movie in allMovies:
    merged.loc[~merged[movie + "_x"].isin([" Sat", " Sun"]), movie] = -1
    merged.loc[merged[movie + "_y"] != " h", movie] = -1
res2 = pd.DataFrame(recomendation(merged[allMovies], data33, 30))
r = np.around(np.mean([rating for rating in list(filter(lambda rating: rating != -1, data33.values[0]))]), 3)
res2 = res2[(res2[1] >= r)].values
result2 = {}
for movie in res2:
    result2.update({movie[0]: movie[1]})

# Запись в json
result = {"user": 33,
          "1": result1,
          "2": result2
          }
with open('result.json', 'w') as f:
    json.dump(result, f, indent=10)
