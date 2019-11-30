import RNBA
import NeuralNetwork
import numpy as np

if __name__ == '__main__':
    player_path = 'nba-enhanced-stats/2012-18_playerBoxScore.csv'

num_games = [1,2,3,4,5,6,7,8,9,10]
num_players = num_games

models = [(10, [500, 100, 50, 20, 20, "f", 10, 10, 10, 10, 1], "NNF_10L_500_100_50_20x2_f_10x4_2_exp", "forked", "exponential"),\
        (10, [500, 100, 50, 20, 20, "f", 10, 10, 10, 10, 149-58+1], "NNF_10L_500_100_50_20x2_f_10x4_149_58_sftmx", "forked", "softmax")]

for i in range(2):
    OutAccd = np.zeros((10,10))
    OutAcct = np.zeros((10,10))
    for j in range(5):
        for k in range(5):
            clf = RNBA.Refridgerator(num_players = num_players[j], num_games = num_games[k])
            clf.Build_Dataset(player_path, Normalized = True)
            X_train, Y_train, X_dev, Y_dev, X_test, Y_test = clf.Read_Dataset(Normalized = True)

            NN = NeuralNetwork.FCNN(models[i][0], models[i][1], models[i][2], num_epochs = 300, output_layer = models[i][4],\
                    Type = models[i][3])
            if models[i][3] == "forked" and models[i][4] == "exponential":
                Y_t1, Y_t2 = clf.Forked(Y_train)
                Y_d1, Y_d2 = clf.Forked(Y_dev)
            if models[i][3] == "forked" and models[i][4] == "softmax":
                Y_t1, Y_t2 = clf.ToSoftmax(Y_train, 58, 149)
                Y_d1, Y_d2 = clf.ToSoftmax(Y_dev, 58, 149)
            model = NN.Model(X_train, Y_t1, Y_t2)
            MSE_train, predictions_train = NN.predict(model, X_train, [Y_t1, Y_t2])
            if models[i][3] == "forked" and models[i][4] == "exponential":
                Y_ptrain = clf.UnFork(predictions_train)
            if models[i][3] == "forked" and models[i][4] == "softmax":
                Y_ptrain = clf.FromSoftmax(predictions_train, 58)
            acc = clf.OutcomeAccuracy(Y_ptrain, Y_train)

            MSE_dev, predictions_dev = NN.predict(model, X_dev, [Y_d1, Y_d2])
            Y_pdev = clf.FromSoftmax(predictions_dev, 58)
            accd = clf.OutcomeAccuracy(Y_pdev, Y_dev)

            OutAccd[j,k] = accd
            OutAcct[j,k] = acc

    np.savetxt("dev_set_acc_"+NN.model_name+".txt", OutAccd)
    np.savetxt("train_set_acc_"+NN.model_name+".txt", OutAcct)

#print(acc)
#print(Y_train)
#print(Y_ptrain)

#print(accd)
#print(Y_dev)
#print(Y_pdev)