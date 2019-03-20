import numpy as np
from sklearn.model_selection import train_test_split
from input import get_train_data, get_eval_data
from keras.models import Sequential
from keras.layers import Dense
from tqdm import tqdm

def main():
    train_data = get_train_data()
    X, y = process_data(train_data)
    train_X, test_X, train_y, test_y = train_test_split(X, y)
    eval_data = get_eval_data()
    eval_X, eval_y = process_data(eval_data)
    model = Sequential()
    model.add(Dense(40, input_dim=len(train_X.columns), activation="relu"))
    model.add(Dense(20))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(train_X, train_y, epochs=10, batch_size=128)
    pred_y = np.round(model.predict(test_X).flatten())
    print(np.unique(pred_y, return_counts=True))
    print("Training accuracy: ", np.sum(pred_y == test_y) / np.size(test_y))
    pred_y = np.round(model.predict(eval_X).flatten())
    print('Eval accuracy:', np.sum(pred_y == eval_y) / np.size(eval_y))

def process_data(data):
    x = data.drop(['Winner', 'Season', 'WTeamID', 'LTeamID'], axis=1)
    y = data['Winner']
    return x, y
 
if __name__=="__main__":
     main()
