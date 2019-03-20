import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from input import get_train_data, get_eval_data
from tqdm import tqdm

def main():
    train_data = get_train_data()
    X, y = process_data(train_data)
    train_X, test_X, train_y, test_y = train_test_split(X, y)
    eval_data = get_eval_data()
    eval_X, eval_y = process_data(eval_data)

    tree = RandomForestClassifier(80) 
    tree.fit(train_X, train_y)
    pred_y = tree.predict(test_X)
    print("Training accuracy: ", np.sum(pred_y == test_y) / np.size(test_y))
    pred_y = tree.predict(eval_X)
    print('Eval accuracy:', np.sum(pred_y == eval_y) / np.size(eval_y))

def process_data(data):
    x = data.drop(['Winner', 'Season', 'WTeamID', 'LTeamID'], axis=1)
    y = data['Winner']
    return x, y
 
if __name__=="__main__":
     main()
