import numpy as np
import RNBA
import csv
import utilities

if __name__ == '__main__':
    player_path = 'nba-enhanced-stats/2012-18_playerBoxScore.csv'

num_players = np.array([1,2,3,4,5,6,7,8,9,10])
num_games = num_players

for i in range(10):
    for j in range(10):
        clf = RNBA.Refridgerator(num_players = num_players[i], num_games = num_games[j])
        clf.Build_Dataset(player_path, Normalized = True)