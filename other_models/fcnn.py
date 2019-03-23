import numpy as np
from sklearn.model_selection import train_test_split
from input import get_train_data, get_eval_data
from keras.models import Sequential
from keras.layers import Dense
from tqdm import tqdm

def main():
    np.random.seed(2410)

    train_data = get_train_data()
    X, y = process_data(train_data)
    train_X, test_X, train_y, test_y = train_test_split(X, y)
    eval_data = get_eval_data()
    eval_X, eval_y = process_data(eval_data)
    model = Sequential()
    model.add(Dense(20, input_dim=len(train_X.columns)))
    model.add(Dense(10))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(train_X, train_y, epochs=10, batch_size=32)
    pred_y = np.round(model.predict(test_X).flatten())
    print(np.unique(pred_y, return_counts=True))
    print("Training accuracy: ", np.sum(pred_y == test_y) / np.size(test_y))
    pred_y = np.round(model.predict(eval_X).flatten())
    print('Eval accuracy:', np.sum(pred_y == eval_y) / np.size(eval_y))

    model.save('model.h5')

def process_data(data):
#    x = data.drop(['Winner', 'Season', 'WTeamID', 'LTeamID', 'T0OrdinalRank', 'T1OrdinalRank'], axis=1)
    x = data[["T0OrdinalRank", "T1OrdinalRank", "T03ptpct", "T0APG", "T0APG_diff", "T0BPG", "T0BPG_diff", "T0DRPG", "T0FPG_diff", "T0PPG_diff", "T0SPG", "T13ptpct", "T1APG", "T1APG_diff", "T1BPG", "T1BPG_diff", "T1DRPG", "T1FPG_diff", "T1PPG_diff", "T1SPG"]]
    y = data['Winner']
    return x, y
 
if __name__=="__main__":
     main()
