import RNBA
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    player_path = 'nba-enhanced-stats/2012-18_playerBoxScore.csv'

num_games = [1,2,3,4,5,6,7,8,9,10]
num_players = num_games

SVMOutAccd = np.zeros((10,10))
SVMOutAcct = np.zeros((10,10))
LROutAccd = np.zeros((10,10))
LROutAcct = np.zeros((10,10))
QDAOutAccd = np.zeros((10,10))
QDAOutAcct = np.zeros((10,10))
for j in [6]:#range(10):
    for k in [4]:#range(10):
        print(num_players[j], num_games[k])
        clf = RNBA.Refridgerator(num_players = num_players[j], num_games = num_games[k])
        clf.Build_Dataset(player_path, Normalized = True)
        X_train, Y_train, X_dev, Y_dev, X_test, Y_test = clf.Binary_Clf_Dataset(normalized = True)

        QDA = QuadraticDiscriminantAnalysis()
        QDA.fit(X_train, Y_train.ravel())
        SVM = SVC(kernel='linear')
        SVM.fit(X_train, Y_train.ravel())
        LR = LogisticRegression()
        LR.fit(X_train, Y_train.ravel())

        Y_pdQDA = QDA.predict(X_dev)
        Y_pdSVM = SVM.predict(X_dev)
        Y_pdLR = LR.predict(X_dev)
        Y_ptQDA = QDA.predict(X_train)
        Y_ptSVM = SVM.predict(X_train)
        Y_ptLR = LR.predict(X_train)

        SVMOutAccd[j,k] = accuracy_score(Y_dev, Y_pdSVM)
        SVMOutAcct[j,k] = accuracy_score(Y_train, Y_ptSVM)
        QDAOutAccd[j,k] = accuracy_score(Y_dev, Y_pdQDA)
        QDAOutAcct[j,k] = accuracy_score(Y_train, Y_ptQDA)
        LROutAccd[j,k] = accuracy_score(Y_dev, Y_pdLR)
        LROutAcct[j,k] = accuracy_score(Y_train, Y_ptLR)

np.savetxt("accuracies/dev_set_acc_SVM.txt", SVMOutAccd)
np.savetxt("accuracies/train_set_acc_SVM.txt", SVMOutAcct)
np.savetxt("accuracies/dev_set_acc_LR.txt", LROutAccd)
np.savetxt("accuracies/train_set_acc_LR.txt", LROutAcct)
np.savetxt("accuracies/dev_set_acc_QDA.txt", QDAOutAccd)
np.savetxt("accuracies/train_set_acc_QDA.txt", QDAOutAcct)
