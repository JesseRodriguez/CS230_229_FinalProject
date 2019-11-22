import RNBA
import NeuralNetwork

if __name__ == '__main__':
    player_path = 'nba-enhanced-stats/2012-18_playerBoxScore.csv'

num_games = [1,2,3,4,5,6,7,8,9,10]
num_players = num_games

models = [(10, [500, 100, 50, 20, 20, 10, 10, 10, 10, 2], "NN_10L_500_100_50_20x2_10x4_2_exp", "stacked", "exponential"),\
        (10, [500, 100, 50, 20, 20, "f", 10, 10, 10, 10, 2], "NNF_10L_500_100_50_20x2_f_10x4_2_exp", "forked", "exponential"),\
        (10, [500, 100, 50, 20, 20, 10, 10, 10, 10, 298], "NN_10L_500_100_50_20x2_10x4_298_sftmx", "stacked", "softmax"),\
        (10, [500, 100, 50, 20, 20, "f", 10, 10, 10, 10, 149], "NNF_10L_500_100_50_20x2_f_10x4_149_sftmx", "forked", "softmax")]