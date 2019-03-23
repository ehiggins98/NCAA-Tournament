This repository contains a few models for predicting results of NCAA tournament games. It was done as the group project for EECS 649 (Introduction to Artificial Intelligence).

The polynomial_regression folder contains code for a 3rd-degree polynomial regression that predicts game scores, written in Python using scikit-learn. The logistic_regression folder contains logistic models written in Python - using scikit-learn - and R. Finally, the other_models folder contains code for a dense neural network classifier, a random forest classifier, and a k-nearest-neighbor classifier. All models were trained on NCAA regular season data and evalutated on NCAA tournament data.

## Accuracy
The polynomial regression model achieved a mean absolute error of 7.83 points on the training set, and 8.20 points on the test set. That is, its predictions were on average around 4 points off from the team's true score. r-squared for this model was 0.265, and when applied to the classification task (predicting winners of games) it achieved 72.1% accuracy.

The logistic regresssion model achieved 73.4% accuracy on the training set (regular-season games) and 68.9% on NCAA tournament games. The R model, though, achieved 72.1% on the training set and 65.3% on the test set.

The neural network achieved 73.66% accuracy on the regular season data and 71.66% on tournament data.

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
* `RPG`: Average rebounds per game. That is, the sum of offensive and defensive rebounds
* `Ordinal`: The team's KenPom rating at the time of the game
* `FPG_diff`: Average foul difference. That is, the average difference between the team's fouls and the opposing team's, computed over every game.
* `BPG_diff`: Average blocks per game difference
* `SPG_diff`: Average steals per game difference
* `APG_diff`: Average assists per game difference
* `ORPG_diff`: Average offensive rebounds per game difference
* `DRPG_diff`: Average defensive rebounds per game difference
* `PPG_diff`: Average points per game difference
* `RPG_diff`: Average rebounds per game difference