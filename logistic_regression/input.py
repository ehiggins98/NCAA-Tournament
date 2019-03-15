import numpy as np
import pandas as pd
import random

def get_train_data():
    try:
        return pd.read_csv('../data/train_data.csv').filter(regex='^(?!Unnamed).+$')
    except FileNotFoundError:
        return parse_data('RegularSeasonCompactResults.csv', 'train_data.csv')

def get_eval_data():
    try:
        return pd.read_csv('../data/eval_data.csv').filter(regex='^(?!Unnamed).+$')
    except FileNotFoundError:
        return parse_data('NCAATourneyCompactResults.csv', 'eval_data.csv')
    
def parse_data(file_name, output_file_name):
    team_data = get_team_data()
    games = pd.read_csv(f'../data/DataFiles/{file_name}')
    games = games.filter(items=['Season', 'WTeamID', 'LTeamID'], axis=1)
    team_data = team_data.drop('Unnamed: 0', axis=1)
    team_data['Season'] = team_data['Season'].astype(int)
    team_data['TeamID'] = team_data['TeamID'].astype(int)
    
    games = games.loc[games['Season'] >= 2003]

    merged = games.merge(team_data, how='left', left_on=['WTeamID', 'Season'], right_on=['TeamID', 'Season'])
    merged = merged.drop('TeamID', axis=1)
    merged = merged.rename(index=str, columns=lambda x: x if x in {'Season', 'WTeamID', 'LTeamID'} else 'T0' + x)

    merged = merged.merge(team_data, how='left', left_on=['LTeamID', 'Season'], right_on=['TeamID', 'Season'])
    merged = merged.drop('TeamID', axis=1)
    merged = merged.rename(index=str, columns=lambda x: x if x in {'Season', 'WTeamID', 'LTeamID'} or x.startswith('T0') else 'T1' + x)

    merged['Winner'] = 0
    merged = merged.apply(randomize, axis=1)
    merged[['Season', 'WTeamID', 'LTeamID', 'Winner']] = merged[['Season', 'WTeamID', 'LTeamID', 'Winner']].astype(int)

    merged.to_csv(f'../data/{output_file_name}')
    return merged

def randomize(row):
    swap = random.random() > 0.5
    print(row['Season'])
    if swap:
        temp = row.filter(regex='^T0.*$')
        row[row.filter(regex='^T0.*$').index] = row.filter(regex='^T1.*$')
        row[row.filter(regex='^T1.*$').index] = temp
        row['Winner'] = 1
    
    return row

def get_team_data():
    try:
        return pd.read_csv('../data/teams_data.csv')
    except FileNotFoundError:
        return parse_team_data()

def parse_team_data():
    games = pd.read_csv('../data/DataFiles/RegularSeasonDetailedResults.csv')
    teams_data = pd.DataFrame()

    print('Parsing season data...')
    
    for i in range(1101, 1467):
        winning_games = process_games(games, i, 'W')
        losing_games = process_games(games, i, 'L')

        team_games = pd.concat([winning_games, losing_games], axis=0)

        print(f'TeamID: {i}')
        
        for season in team_games['Season'].unique():
            season_games = team_games.loc[team_games['Season'] == season]
            num_games = season_games.count()
            all_columns = set(season_games)
            exclude_columns = {'Season', 'TeamID'}
            season_data = pd.concat([season_games[list(all_columns - exclude_columns)].sum(axis=0), season_games[list(exclude_columns)].mean().astype(int)], axis=0)

            season_data = pd.DataFrame(
                data=
                {
                    'TeamID': season_data['TeamID'], # team id
                    'Season': season_data['Season'], # season
                    '2ptpct': (season_data['FGM'] - season_data['FGM3']) / (season_data['FGA'] - season_data['FGA3']), # 2 pt percentage
                    '3ptpct': season_data['FGM3'] / season_data['FGA3'], # 3 point percentage
                    'FTpct': season_data['FTM'] / season_data['FTA'], # free throw percentage
                    'FPG': season_data['PF'] / num_games, # fouls per game
                    'BPG': season_data['Blk'] / num_games, # blocks per game
                    'SPG': season_data['Stl'] / num_games, # steals per game
                    'APG': season_data['Ast'] / num_games, # assists per game
                    'ORPG': season_data['OR'] / num_games, # offensive rebounds per game
                    'DRPG': season_data['DR'] / num_games, # defensive rebounds per game
                    'PPG': season_data['Score'] / num_games # points per game
                }
            )

            teams_data = teams_data.append(season_data.loc['Season'], ignore_index=True)
    teams_data.to_csv('../data/teams_data.csv')
    return teams_data

def process_games(games, team_id, w_or_l):
    relevant_games = games.loc[games[f'{w_or_l.upper()}TeamID'] == team_id]
    relevant_games = relevant_games.filter(regex=f'Season|^{w_or_l.upper()}.*|WLoc')
    relevant_games = relevant_games.rename(index=str, columns=lambda x: x if x == 'Season' else x[1:])
    relevant_games = relevant_games.drop(labels='Loc', axis=1)

    return relevant_games
