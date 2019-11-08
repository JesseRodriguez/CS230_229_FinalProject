import RNBA
import NeuralNetwork


if __name__ == '__main__':
    player_path = 'nba-enhanced-stats/2012-18_playerBoxScore.csv'

clf = RNBA.Refridgerator(num_players = 5, num_games = 5)
clf.Build_Dataset(player_path, Normalized = True)
X_train, Y_train, X_dev, Y_dev, X_test, Y_test = clf.Read_Dataset(Normalized = True)

NN = NeuralNetwork.FCNN(3, [3,3,2], "NN_3L_332")
parameters = NN.model(X_train, Y_train)
dpredictions = NN.predict(X_dev, parameters)
print(type(dpredictions))