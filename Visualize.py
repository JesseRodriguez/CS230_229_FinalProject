import utilities
import RNBA
import numpy as np

clf = RNBA.Refridgerator(num_players = 1, num_games = 1)
clf.Build_Dataset('nba-enhanced-stats/2012-18_playerBoxScore.csv', Normalized = True)
X_train, Y_train, X_dev, Y_dev, X_test, Y_test = clf.Read_Dataset(Normalized = True)

_, Y_tr = clf.OnePerEx(X_train, Y_train)
_, Y_d = clf.OnePerEx(X_dev, Y_dev)
_, Y_te = clf.OnePerEx(X_test, Y_test)

xlabels = ["Individual Team Score", "Combined Team Score", "Point Spread"]
savepaths = ["IndividualScoreHist.pdf", "CombinedScoreHist.pdf", "SpreadHist.pdf"]

for i in range(3):
    if i == 2:
        data_tr = clf.Spreads(Y_train)
        data_d = clf.Spreads(Y_dev)
        data_te = clf.Spreads(Y_test)
    elif i == 1:
        data_tr = clf.CombinedScores(Y_train)
        data_d = clf.CombinedScores(Y_dev)
        data_te = clf.CombinedScores(Y_test)
    else:
        data_tr = Y_tr
        data_d = Y_d
        data_te = Y_te

    utilities.histogram(data_tr, xlabels[i], "Train Set", savepath = "plots/train_"+savepaths[i])
    utilities.histogram(data_d, xlabels[i], "Dev Set", savepath = "plots/dev_"+savepaths[i])
    utilities.histogram(data_te, xlabels[i], "Test Set", savepath = "plots/test_"+savepaths[i])
    utilities.stackedhist(data_tr, data_d, data_te, xlabels[i], ["Train", "Dev", "Test"],\
            savepath = "plots/comparison_"+savepaths[i])

