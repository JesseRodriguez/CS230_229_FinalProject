import RNBA
import NeuralNetwork
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score
import tensorflow.keras.backend as K
import utilities

if __name__ == '__main__':
    player_path = 'nba-enhanced-stats/2012-18_playerBoxScore.csv'

#Build Relevant Datasets
QDAclf = RNBA.Refridgerator(num_players = 10, num_games = 5)
QDAclf.Build_Dataset(player_path, Normalized = True)
X_train_QDA, Y_train_QDA, X_dev_QDA, Y_dev_QDA, X_test_QDA, Y_test_QDA = QDAclf.Binary_Clf_Dataset(normalized = True)

SVMclf = RNBA.Refridgerator(num_players = 3, num_games = 3)
SVMclf.Build_Dataset(player_path, Normalized = True)
X_train_SVM, Y_train_SVM, X_dev_SVM, Y_dev_SVM, X_test_SVM, Y_test_SVM = SVMclf.Binary_Clf_Dataset(normalized = True)

LRclf = RNBA.Refridgerator(num_players = 7, num_games = 5)
LRclf.Build_Dataset(player_path, Normalized = True)
X_train_LR, Y_train_LR, X_dev_LR, Y_dev_LR, X_test_LR, Y_test_LR = LRclf.Binary_Clf_Dataset(normalized = True)

NNEclf = RNBA.Refridgerator(num_players = 3, num_games = 5)
NNEclf.Build_Dataset(player_path, Normalized = True)
X_train_NNE, Y_train_NNE, X_dev_NNE, Y_dev_NNE, X_test_NNE, Y_test_NNE = NNEclf.Read_Dataset(Normalized = True)

NNSclf = RNBA.Refridgerator(num_players = 2, num_games = 4)
NNSclf.Build_Dataset(player_path, Normalized = True)
X_train_NNS, Y_train_NNS, X_dev_NNS, Y_dev_NNS, X_test_NNS, Y_test_NNS = NNSclf.Read_Dataset(Normalized = True)

NNBclf = RNBA.Refridgerator(num_players = 3, num_games = 4)
NNBclf.Build_Dataset(player_path, Normalized = True)
X_train_NNB, Y_train_NNB, X_dev_NNB, Y_dev_NNB, X_test_NNB, Y_test_NNB = NNBclf.Binary_Clf_Dataset(normalized = True)

#Train Models
QDA = QuadraticDiscriminantAnalysis()
QDA.fit(X_train_QDA, Y_train_QDA.ravel())

SVM = SVC(kernel='linear')
SVM.fit(X_train_SVM, Y_train_SVM.ravel())

LR = LogisticRegression()
LR.fit(X_train_LR, Y_train_LR.ravel())

NNE = NeuralNetwork.FCNN(6, [6,5,5,"f",5,5,1], "ExpTest", num_epochs = 400, output_layer = "exponential",\
        Type = "forked", Regularized = None)
Y_te1, Y_te2 = NNEclf.Forked(Y_train_NNE)
Y_tee1, Y_tee2 = NNEclf.Forked(Y_test_NNE)
modelE = NNE.Model(X_train_NNE, Y_te1, Y_te2)
Loss_test_NNE, pred_test_NNE = NNE.predict(modelE, X_test_NNE, [Y_tee1, Y_tee2])
Y_ptest_NNE = NNEclf.UnFork(pred_test_NNE)
NNEOutAccte = NNEclf.OutcomeAccuracy(Y_ptest_NNE, Y_test_NNE)
K.clear_session()

NNS = NeuralNetwork.FCNN(9, [200,100,50,20,10,5,"f",5,5,149-58+1], "SoftTest", num_epochs = 400, output_layer = "softmax",\
        Type = "forked", Regularized = None)
Y_ts1, Y_ts2 = NNSclf.ToSoftmax(Y_train_NNS, 58, 149)
Y_tes1, Y_tes2 = NNSclf.ToSoftmax(Y_test_NNS, 58, 149)
modelS = NNS.Model(X_train_NNS, Y_ts1, Y_ts2)
Loss_test_NNS, pred_test_NNS = NNS.predict(modelS, X_test_NNS, [Y_tes1, Y_tes2])
Y_ptest_NNS = NNSclf.FromSoftmax(pred_test_NNS, 58)
NNSOutAccte = NNSclf.OutcomeAccuracy(Y_ptest_NNS, Y_test_NNS)
K.clear_session()

NNB = NeuralNetwork.FCNN(4, [2,5,5,1], "BinTest", num_epochs = 75, output_layer = "sigmoid",\
        Type = "stacked", Regularized = None)
modelB = NNB.Model(X_train_NNB, Y_train_NNB)
Loss_test_NNB, pred_test_NNB = NNB.predict(modelB, X_test_NNB, Y_test_NNB)
NNBOutAccte = Loss_test_NNB[1]
K.clear_session()

#Get Predictions
Y_pteQDA = QDA.predict(X_test_QDA)
Y_pteSVM = SVM.predict(X_test_SVM)
Y_pteLR = LR.predict(X_test_LR)

#Outcome Accuracies
SVMOutAccte = accuracy_score(Y_test_SVM, Y_pteSVM)
QDAOutAccte = accuracy_score(Y_test_QDA, Y_pteQDA)
LROutAccte = accuracy_score(Y_test_LR, Y_pteLR)

print(LROutAccte, SVMOutAccte, QDAOutAccte, NNBOutAccte, NNSOutAccte, NNEOutAccte)

_, Y_test = NNSclf.OnePerEx(X_test_NNS, Y_test_NNS)
_, Y_pred_NNS = NNSclf.OnePerEx(X_test_NNS, Y_ptest_NNS)
_, Y_pred_NNE = NNSclf.OnePerEx(X_test_NNE, Y_ptest_NNE)

xlabels = ["Individual Team Score", "Combined Team Score", "Point Spread"]
savepaths = ["IndividualScoreHist.pdf", "CombinedScoreHist.pdf", "SpreadHist.pdf"]

for i in range(3):
    if i == 2:
        data_te = NNSclf.Spreads(Y_test_NNS)
        data_teNNS = NNSclf.Spreads(Y_ptest_NNS)
        data_teNNE = NNEclf.Spreads(Y_ptest_NNE)
    elif i == 1:
        data_te = NNSclf.CombinedScores(Y_test_NNS)
        data_teNNS = NNSclf.CombinedScores(Y_ptest_NNS)
        data_teNNE = NNEclf.CombinedScores(Y_ptest_NNE)
    else:
        data_te = Y_test
        data_teNNS = Y_pred_NNS
        data_teNNE = Y_pred_NNE

    utilities.histogram(data_te, xlabels[i], "Test Set", savepath = "plots/test_"+savepaths[i])
    utilities.histogram(data_teNNS, xlabels[i], "Softmax Predictions", savepath = "plots/sftmxPred_"+savepaths[i])
    utilities.histogram(data_teNNE, xlabels[i], "Exponential Predictions", savepath = "plots/expPred_"+savepaths[i])
    utilities.stackedhist(data_te, data_teNNS, data_teNNE, xlabels[i], ["Test Set", "Softmax", "Exponential"],\
            savepath = "plots/Pred_comparison_"+savepaths[i])


