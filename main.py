from sklearn.naive_bayes import MultinomialNB
import numpy as np
from operator import itemgetter
import random

with open('Programmatic Project_Scoring_TTD pixel fires.csv', encoding="Latin-1") as input_file:
    cookies = {}
    scores = {}
    scores_ = {}
    countries = {}
    regions = {}
    devices = {}
    op_sys = {}

    contact_id = "qelg9wq"
    header = next(input_file, None)  # skip the headers
    i = 0
    for row in input_file:
        row = row.split(",")
        if row[0] in cookies:
            cookie = cookies[row[0]]
            cookie[0] = cookie[0] or row[1] == contact_id
        else:
            if row[3] not in countries:
                countries[row[3]] = len(countries) + 1
            if row[4] not in regions:
                regions[row[4]] = len(regions) + 1
            if row[10] not in devices:
                devices[row[10]] = len(devices) + 1
            if row[11] not in op_sys:
                op_sys[row[11]] = len(op_sys) + 1
            cookies[row[0]] = [row[1] == contact_id, countries[row[3]], regions[row[4]], devices[row[10]], op_sys[row[11]]]

    clf = MultinomialNB()
    training_keys = random.sample(list(cookies.keys()), 10000)
    training_cookies = [cookies[key] for key in training_keys]
    training_x = np.array([cookie[1:] for cookie in training_cookies])
    training_y = np.array([cookie[0] for cookie in training_cookies])
    clf.fit(training_x, training_y)
    for tdid in cookies:
        cookie = cookies[tdid]
        x = np.array([cookie[1:]])
        scores[tdid] = clf.predict_proba(x)[0][1]
    input_file.seek(0)
    next(input_file, None)  # skip the headers
    output_file = open("output.csv", 'w', encoding="Latin-1")
    output_file.write(header)
    rows = []
    for row in input_file:
        row = row.split(",")[:-1]
        row.append(int(scores[row[0]] * 10000) / 100.)
        rows.append(row)
    rows = sorted(rows, key=itemgetter(15))
    rows.reverse()
    output_file.writelines([",".join([str(item) for item in row]) + "\n" for row in rows])
