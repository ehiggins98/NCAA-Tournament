from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from input import get_train_data, get_eval_data
import numpy as np
import pandas as pd

def main():
    model = LinearRegression()
    train_data = get_train_data()
    train_x, train_y = process_data(train_data)
    train_x = pd.concat([train_x, np.square(train_x), np.power(train_x, 3)], axis=1)
    
    model.fit(train_x, train_y)
    pred_y = np.round(model.predict(train_x))
    print('Train mean absolute error:', np.sum(np.absolute(pred_y - train_y)) / np.size(pred_y))

    eval_data = get_eval_data()
    eval_x, eval_y = process_data(eval_data)
    eval_x = pd.concat([eval_x, np.square(eval_x), np.power(eval_x, 3)], axis=1)
    pred_y = np.round(model.predict(eval_x))
    print('Eval mean absolute error:', np.sum(np.absolute(pred_y - eval_y))/ np.size(eval_y))

    df = pd.DataFrame({'Score1': pred_y[:int(pred_y.size/2)], 'Score2': pred_y[int(pred_y.size/2):]})
    df['Winner'] = df['Score1'] < df['Score2']
    print(np.sum(df['Winner'] == eval_data['Winner']) / np.size(df['Winner']))
    print(r2_score(eval_y, pred_y))

def process_data(data):
    x = data.drop(['Winner', 'Season', 'WTeamID', 'LTeamID', 'Winner', 'T0Score', 'T1Score'], axis=1)
    x2 = x.copy()

    filteredT0 = list(filter(lambda x: x.startswith('T0'), x2.columns))
    filteredT1 = list(filter(lambda x: x.startswith('T1'), x2.columns))

    temp = x2[filteredT0].copy()
    x2[filteredT0] = x2[filteredT1].values
    x2[filteredT1] = temp.values

    y1 = data['T0Score']
    y2 = data['T1Score'].rename('T0Score')

    return pd.concat([x, x2], ignore_index=True), pd.concat([y1, y2], ignore_index=True)

def lookup_teams(data):
    teams = pd.read_csv('../data/DataFiles/Teams.csv')
    data = data.merge(teams, left_on='WTeamID', right_on='TeamID')
    data = data.rename({'TeamName': 'T0TeamName'}, axis=1)

    data = data.merge(teams, left_on='LTeamID', right_on='TeamID')
    data = data.rename({'TeamName': 'T1TeamName'}, axis=1)

    return data

if __name__ == '__main__':
    main()