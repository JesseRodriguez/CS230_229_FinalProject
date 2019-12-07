import RNBA
import NeuralNetwork
import numpy as np
import tensorflow.keras.backend as K

if __name__ == '__main__':
    player_path = 'nba-enhanced-stats/2012-18_playerBoxScore.csv'

num_games = [1,2,3,4,5,6,7,8,9,10]
num_players = num_games

models = [(6, [6, 5, 5, "f", 5, 5, 1], "NNF_10L_5x5_f_5x4_2_exp", "forked", "exponential", None),\
        #[(5, [5, 5, "f", 5, 5, 1], "NNF_10L_5x5_f_5x4_2_exp", "forked", "exponential", None),\
        #(9, [200, 100, 50, 20, 10, 5, "f", 5, 5, 149-58+1], "NNF_10L_5x5_f_5x4_149_58_sftmx", "forked", "softmax", None),
        (9, [200, 100, 50, 20, 10, 5, "f", 5, 5, 149-58+1], "NNF_10L_5x5_f_5x4_149_58_sftmx", "forked", "softmax", None),
        #(4, [2,5,5,1], "NNS_6L_5x5_1_sig", "stacked", "sigmoid", "L2")]
        (4, [2,5,5,1], "NNS_6L_5x5_1_sig", "stacked", "sigmoid", None)]

for i in [0]:#range(2):
    OutAccd = np.zeros((10,10))
    OutAcct = np.zeros((10,10))
    for j in range(10):#[1,5,9]:#range(1):
        for k in range(10):#[1,5,9]:#range(1):
            clf = RNBA.Refridgerator(num_players = num_players[j], num_games = num_games[k])
            clf.Build_Dataset(player_path, Normalized = True)
            X_train, Y_train, X_dev, Y_dev, X_test, Y_test = clf.Read_Dataset(Normalized = True)

            NN = NeuralNetwork.FCNN(models[i][0], models[i][1], models[i][2], num_epochs = 400, output_layer = models[i][4],\
                    Type = models[i][3], Regularized = models[i][5])
            if models[i][3] == "forked" and models[i][4] == "exponential":
                Y_t1, Y_t2 = clf.Forked(Y_train)
                Y_d1, Y_d2 = clf.Forked(Y_dev)
            if models[i][3] == "forked" and models[i][4] == "softmax":
                Y_t1, Y_t2 = clf.ToSoftmax(Y_train, 58, 149)
                Y_d1, Y_d2 = clf.ToSoftmax(Y_dev, 58, 149)
            if models[i][4] == "sigmoid":
                _, Y_tb, _, Y_db, _, Y_teb = clf.Binary_Clf_Dataset(normalized = True)
            if models[i][3] == "forked":
                model = NN.Model(X_train, Y_t1, Y_t2)
                Loss_train, predictions_train = NN.predict(model, X_train, [Y_t1, Y_t2])
            elif models[i][3] == "stacked":
                model = NN.Model(X_train, Y_tb)
                Loss_train, predictions_train = NN.predict(model, X_train, Y_tb)
                acc = Loss_train[1]
                Loss_dev, predictions_dev = NN.predict(model, X_dev, Y_db)
                accd = Loss_dev[1]
            if models[i][3] == "forked" and models[i][4] == "exponential":
                Y_ptrain = clf.UnFork(predictions_train)
                acc = clf.OutcomeAccuracy(Y_ptrain, Y_train)
            if models[i][3] == "forked" and models[i][4] == "softmax":
                Y_ptrain = clf.FromSoftmax(predictions_train, 58)
                acc = clf.OutcomeAccuracy(Y_ptrain, Y_train)
            if models[i][3] == "forked":
                MSE_dev, predictions_dev = NN.predict(model, X_dev, [Y_d1, Y_d2])
                if models[i][4] == "softmax":
                    Y_pdev = clf.FromSoftmax(predictions_dev, 58)
                if models[i][4] == "exponential":
                    Y_pdev = clf.UnFork(predictions_dev)
                accd = clf.OutcomeAccuracy(Y_pdev, Y_dev)

            K.clear_session()
            OutAccd[j,k] = accd
            OutAcct[j,k] = acc

    np.savetxt("accuracies/dev_set_acc_"+NN.model_name+".txt", OutAccd)
    np.savetxt("accuracies/train_set_acc_"+NN.model_name+".txt", OutAcct)

#print(acc)
#print(Y_train)
#print(Y_ptrain)

#print(accd)
#print(Y_dev)
#print(Y_pdev)