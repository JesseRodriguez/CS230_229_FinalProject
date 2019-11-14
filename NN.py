import RNBA
import NeuralNetwork


if __name__ == '__main__':
    player_path = 'nba-enhanced-stats/2012-18_playerBoxScore.csv'

clf = RNBA.Refridgerator(num_players = 5, num_games = 5)
clf.Build_Dataset(player_path, Normalized = True)
X_train, Y_train, X_dev, Y_dev, X_test, Y_test = clf.Read_Dataset(Normalized = True)

NN = NeuralNetwork.FCNN(7, [60,50,40,30,20,10,1], "NN_7L_60_50_40_30_20_10_1", num_epochs = 300)

X_train_n, Y_train_n = clf.OnePerEx(X_train, Y_train)
X_dev_n, Y_dev_n = clf.OnePerEx(X_dev, Y_dev)

model = NN.Model(X_train_n, Y_train_n)
MSEt, predictionst = NN.predict(model, X_train_n, Y_train_n)
MSE, predictions = NN.predict(model, X_dev_n, Y_dev_n)

print(MSEt)
print(predictionst)
print(Y_train_n)
print(MSE)
print(predictions)
print(Y_dev_n)