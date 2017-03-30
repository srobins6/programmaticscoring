import csv
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import random

with open('Programmatic Project_Scoring_TTD pixel fires.csv', encoding="Latin-1") as input_file:
    csv_reader = csv.reader(input_file, delimiter=',', quotechar='|')

    cookies = {}
    scores = {}
    countries = {'(null': None}
    regions = {'(null': None}
    cities = {'(null': None}
    zips = {'(null': None}
    devicee = {'(null': None}
    op_sys = {'(null': None}
    contact_id = "qelg9wq"
    next(csv_reader, None)  # skip the headers
    i = 0
    for row in csv_reader:
        if row[0] in cookies:
            cookie = cookies[row[0]]
            cookie[0] = cookie[0] or row[1] == contact_id
        else:
            if row[3] not in countries:
                countries[row[3]] = len(countries) + 1
            if row[4] not in regions:
                regions[row[4]] = len(regions) + 1
            if row[6] not in cities:
                cities[row[6]] = len(cities) + 1
            if row[7] not in zips:
                zips[row[7]] = len(zips) + 1
            if row[10] not in devicee:
                devicee[row[10]] = len(devicee) + 1
            if row[11] not in op_sys:
                op_sys[row[11]] = len(op_sys) + 1
            cookies[row[0]] = [row[1] == contact_id, countries[row[3]], regions[row[4]], cities[row[6]], zips[row[7]], devicee[row[10]], op_sys[row[11]]]
    clf = MultinomialNB()
    training_keys = list(cookies.keys())[0:14262]
    # todo: change this to random sample
    training_cookies = [cookies[key] for key in training_keys]
    training_x = np.array([cookie[1:] for cookie in training_cookies])
    training_y = np.array([cookie[0] for cookie in training_cookies])
    clf.fit(training_x, training_y)

    for tdid in cookies:
        cookie = cookies[tdid]
        x = np.array([cookie[1:]])
        scores[tdid] = clf.predict_proba(x)[0][1]
