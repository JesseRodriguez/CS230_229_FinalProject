import numpy as np
import RNBA
import LogisticRegression
import csv
import utilities

if __name__ == '__main__':
    player_path = 'nba-enhanced-stats/2012-18_playerBoxScore.csv'
    team_path = 'nba-enhanced-stats/2012-18_teamBoxScore.csv'

clf = RNBA.Refridgerator(num_players = 5, num_games = 5)
clf.Build_Dataset(player_path, Normalized = True)
clf.Binary_Clf_Dataset(normalized = True)
X_train, Y_train, X_dev, Y_dev, X_test, Y_test = clf.Read_Dataset(Binary = True, Normalized = True)

LG = LogisticRegression.logreg(L_rate = 0.00003)
cost = LG.fit(X_train, Y_train)
predictions = LG.predict(X_dev)
tpredictions = LG.predict(X_test)
trpredictions = LG.predict(X_train)
with open('theta.csv', mode = 'w') as sfile:
    writer = csv.writer(sfile)
    writer.writerow((LG.bias,0))
    for i in range(np.shape(LG.theta)[0]):
        writer.writerow((LG.theta[i,0],0))

utilities.plot(cost)
print(1-np.sum(np.abs(Y_train-trpredictions))/np.shape(Y_train)[0])
print(1-np.sum(np.abs(Y_dev-predictions))/np.shape(Y_dev)[0])
print(1-np.sum(np.abs(Y_test-tpredictions))/np.shape(Y_test)[0])


