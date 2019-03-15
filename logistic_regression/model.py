from sklearn.linear_model import LogisticRegression
from input import get_train_data, get_eval_data
import numpy as np

def main():
    model = LogisticRegression(penalty='l2')
    train_data = get_train_data()
    train_x, train_y = process_data(train_data)
    
    model.fit(train_x, train_y)
    pred_y = model.predict(train_x)
    print('Train accuracy:', np.sum(pred_y == train_y) / np.size(train_y))

    eval_data = get_eval_data()
    eval_x, eval_y = process_data(eval_data)
    pred_y = model.predict(eval_x)
    print('Eval accuracy:', np.sum(pred_y == eval_y) / np.size(eval_y))

def process_data(data):
    x = data.drop(['Winner', 'Season', 'WTeamID', 'LTeamID'], axis=1)
    y = data['Winner']
    return x, y

if __name__ == '__main__':
    main()