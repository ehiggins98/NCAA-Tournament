This repository contains a few models for predicting results of NCAA tournament games. It was done as the group project for EECS 649 (Introduction to Artificial Intelligence).

So far, the logistic_regression folder contains logistic models written in Python - using scikit-learn - and R. NCAA regular season data was used to train the model, and it was evaluated on NCAA tournament data (see "Data" section below).

## Accuracy
The scikit-learn model achieved 71.8% accuracy on the training set (regular-season games) and 67.7% on NCAA tournament games. The R model, though, achieved 70.0% on the training set and 67.7% on the test set.

## Data
Data was taken from the [NCAA Tournament prediction competition on Kaggle](https://www.kaggle.com/c/mens-machine-learning-competition-2019) and the following columns were extracted from the box score datasets:

* `TeamID`: The integer ID of the team from the dataset
* `Season`: The season (i.e. 2008)
* `2ptpct`: The percentage of two-point shots made by the team
* `3ptpct`: The percentage of three-point shots made by the team
* `FTpct`: The team's free-throw percentage
* `FPG`: Average team fouls per game
* `BPG`: Average blocks per game
* `SPG`: Average steals per game
* `APG`: Average assists per game
* `ORPG`: Average offensive rebounds per game
* `DRPG`: Average defensive rebounds per game
* `PPG`: Average points per game
* `Ordinal`: The team's KenPom rating at the time of the game