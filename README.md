This repository contains a few models for predicting results of NCAA tournament games. It was done as the group project for EECS 649 (Introduction to Artificial Intelligence).

So far, the logistic_regression folder contains logistic models written in Python - using scikit-learn - and R. NCAA regular season data was used to train the model, and it was evaluated on NCAA tournament data (see "Data" section below).

## Accuracy
The scikit-learn model achieved 73.4% accuracy on the training set (regular-season games) and 68.9% on NCAA tournament games. The R model, though, achieved 72.1% on the training set and 65.3% on the test set.

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