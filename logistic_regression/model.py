from sklearn.linear_model import LogisticRegression
from input import get_train_data
import numpy as np

def main():
    model = LogisticRegression(penalty='l2')
    train_data = get_train_data()

    x = train_data.drop(['Winner', 'Season', 'WTeamID', 'LTeamID'], axis=1)
    y = train_data['Winner']
    
    model.fit(x, y)
    pred_y = model.predict(x)
    print('Accuracy:', np.sum(pred_y == y) / np.size(y))

if __name__ == '__main__':
    main()