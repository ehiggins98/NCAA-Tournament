from sklearn.linear_model import LogisticRegression
from input import get_train_data, get_eval_data
import numpy as np
import pandas as pd

def main():
    model = LogisticRegression(penalty='l2', solver='liblinear')
    train_data = get_train_data()
    train_x, train_y = process_data(train_data)
    
    model.fit(train_x, train_y)
    pred_y = model.predict(train_x)
    print('Train accuracy:', np.sum(pred_y == train_y) / np.size(train_y))

    eval_data = get_eval_data()
    eval_x, eval_y = process_data(eval_data)
    pred_y = model.predict(eval_x)
    print('Eval accuracy:', np.sum(pred_y == eval_y) / np.size(eval_y))
    print('\n')

    demo_data = pd.concat([eval_data.loc[eval_data['WTeamID'] == 1242].loc[eval_data['Season'] == 2018], eval_data.loc[eval_data['LTeamID'] == 1242].loc[eval_data['Season'] == 2018]], axis=0)

    demo_x, demo_y = process_data(demo_data)
    pred_y = model.predict(demo_x)
    pred_y = pd.DataFrame(data=pred_y, columns=["Predicted"])

    demo_data = lookup_teams(demo_data)

    result = pd.concat([demo_data, pred_y], axis=1)
    print(result[['Season', 'T0TeamName', 'T1TeamName', 'Winner', 'Predicted']])

def process_data(data):
    x = data.drop(['Winner', 'Season', 'WTeamID', 'LTeamID'], axis=1)
    y = data['Winner']
    return x, y

def lookup_teams(data):
    teams = pd.read_csv('../data/DataFiles/Teams.csv')
    data = data.merge(teams, left_on='WTeamID', right_on='TeamID')
    data = data.rename({'TeamName': 'T0TeamName'}, axis=1)

    data = data.merge(teams, left_on='LTeamID', right_on='TeamID')
    data = data.rename({'TeamName': 'T1TeamName'}, axis=1)

    return data

if __name__ == '__main__':
    main()