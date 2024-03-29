import csv
import pandas as pd
import numpy as np
import os.path
import random
import string

class Refridgerator:
    def __init__(self, num_players = 8, num_games = 5, rank_scheme = "Pts", stats_omitted = []):
        """
        Args:
            num_players: The number of players from each team who's statlines
            will be included in the feature vectors
            num_games: Number of games prior to game being predicted that will
            be included in the feature vector.
            rank_scheme: The metric by which the players are ordered in the
            feature vectors. e.g. if "Pts" is selected, the top scoring
            player by total points is listed first, second highest scorer
            second, and so on. Possible choices are:
                "Pts",
            stats_omitted: The stats from the player's box score that are not
            included in the feature vector.
        """
        self.num_players = num_players
        self.num_games = num_games
        self.rank_scheme = rank_scheme
        self.stats_omitted = stats_omitted
        self.X = None
        self.Y = None
        self.savepath = None
        
    def Read_Dataset(self, Binary = False, Normalized = False):
        """
        Reads existing train/dev/test sets and returns them as numpy arrays
        """
        if Binary:
            Blabel = "binary"
        else:
            Blabel = ""

        if Normalized:
            Nlabel = "normalized"
        else:
            Nlabel = ""
            
        train_data = pd.read_csv('datasets/'+Nlabel+Blabel+'_train_'+self.savepath+'np'+\
                str(self.num_players)+'ng'+str(self.num_games)+'rs'+\
                str(self.rank_scheme)+'so'+str(len(self.stats_omitted))+'.csv',\
                header = None).to_numpy()
        test_data = pd.read_csv('datasets/'+Nlabel+Blabel+'_test_'+self.savepath+'np'+\
                str(self.num_players)+'ng'+str(self.num_games)+'rs'+\
                str(self.rank_scheme)+'so'+str(len(self.stats_omitted))+'.csv',\
                header = None).to_numpy()
        dev_data = pd.read_csv('datasets/'+Nlabel+Blabel+'_dev_'+self.savepath+'np'+\
                str(self.num_players)+'ng'+str(self.num_games)+'rs'+\
                str(self.rank_scheme)+'so'+str(len(self.stats_omitted))+'.csv',\
                header = None).to_numpy()
        
        if Binary:
            X_train = train_data[:,1:]
            Y_train = train_data[:,0]
            X_test = test_data[:,1:]
            Y_test = test_data[:,0]
            X_dev = dev_data[:,1:]
            Y_dev = dev_data[:,0]
        else:
            X_train = train_data[:,2:]
            Y_train = train_data[:,0:2]
            X_test = test_data[:,2:]
            Y_test = test_data[:,0:2]
            X_dev = dev_data[:,2:]
            Y_dev = dev_data[:,0:2]
        
        return X_train, Y_train, X_dev, Y_dev, X_test, Y_test

    def Binary_Clf_Dataset(self, normalized = False):
        """
        Builds Train/Dev/Test sets for Binary classifier model
        """
        if normalized:
            Nlabel = "normalized"
        else:
            Nlabel = ""

        X_train, Y_train, X_dev, Y_dev, X_test, Y_test = self.Read_Dataset(Normalized = normalized)
        Y_train_b = np.ones((np.shape(Y_train)[0],1))
        Y_dev_b = np.ones((np.shape(Y_dev)[0],1))
        Y_test_b = np.ones((np.shape(Y_test)[0],1))

        for i in range(np.shape(X_train)[0]):
            if Y_train[i,0] < Y_train[i,1]:
                Y_train_b[i,0] = 0
        for i in range(np.shape(X_dev)[0]):
            if Y_dev[i,0] < Y_dev[i,1]:
                Y_dev_b[i,0] = 0
        for i in range(np.shape(X_test)[0]):
            if Y_test[i,0] < Y_test[i,1]:
                Y_test_b[i,0] = 0

        return X_train, Y_train_b, X_dev, Y_dev_b, X_test, Y_test_b

    def Forked(self, Y):
        """
        Creates two label arrays for dual output forked NN.
        """
        Y1 = np.reshape(Y[:,0], (Y.shape[0],1))
        Y2 = np.reshape(Y[:,1], (Y.shape[0],1))

        return Y1, Y2

    def UnFork(self, Forked_pred):
        """
        Remerges label vectors to original format to allow fo evaluation
        """
        Y1 = Forked_pred[0]
        Y2 = Forked_pred[1]
        Y = np.append(Y1, Y2, axis = 1)

        return Y

    def ToSoftmax(self, Y, Min, Max):
        """
        Creates one-hot label vectors for softmax output layer
        only works with forked models
        """
        n = Y.shape[0]
        Y1 = np.zeros((n, Max-Min+1))
        Y2 = np.zeros((n, Max-Min+1))
        for i in range(n):
            Y1[i,int(Y[i,0])-Min] = 1
            Y2[i,int(Y[i,1])-Min] = 1

        return Y1, Y2

    def FromSoftmax(self, Forked_pred, Min):
        """
        Transforms softmax predictions into the prediction format for evaluation
        """
        Y1 = Forked_pred[0]
        Y2 = Forked_pred[1]
        n = Y1.shape[0]
        Yp = np.zeros((n,2))
        for i in range(n):
            s1 = np.argmax(Y1[i,:]) + Min
            s2 = np.argmax(Y2[i,:]) + Min
            Yp[i,0] = s1
            Yp[i,1] = s2

        return Yp

    def OnePerEx(self, X, Y):
        """
        Turns data into smaller feature vectors and labels, where each x is just
        one team's stats, and each label y is the corresponding score
        """
        half = X.shape[1]//2
        fhalf = X[:,:half]
        lhalf = X[:,half:]
        X_n = np.append(fhalf, lhalf, axis = 0)
        Y_n = np.append(Y[:,0],Y[:,1])

        return X_n, Y_n

    def OutcomeAccuracy(self, Y_p, Y_t, OnePer = False):
        """
        Takes predictions and labels and returns the pecentage of game outcomes
        that were predicted correctly
        """
        if OnePer:
            half = Y_p.shape[0]//2
            Y_p = np.append(Y_p[:half].reshape((half,1)),Y_p[half:].reshape((half,1)), axis = 1)
            Y_t = np.append(Y_t[:half].reshape((half,1)),Y_t[half:].reshape((half,1)), axis = 1)

        correct = 0
        for i in range(Y_p.shape[0]):
            if Y_p[i,0] > Y_p[i,1]:
                winp = 1
            else:
                winp = 0
            if Y_t[i,0] > Y_t[i,1]:
                wint = 1
            else:
                wint = 0
            if winp == wint:
                correct += 1
        acc = correct/Y_p.shape[0]

        return acc

    def BinaryAccuracy(self, Y_p, Y_t):
        """
        Takes predictions from binary classifier and outputs accuracy
        """
        n = Y_t.shape[0]
        Y_p = np.reshape(Y_p, Y_t.shape)

        Acc = np.sum(1-np.abs(Y_p-Y_t))/n

        return Acc

    def Spreads(self, Y):
        """
        Takes arrray of labels and outputs the spread for each game
        """
        n = Y.shape[0]
        Spreads = np.zeros((n,1))
        for i in range(n):
            Spreads[i] = abs(Y[i,0]-Y[i,1])

        return Spreads

    def CombinedScores(self, Y):
        """
        Takes arrray of labels and outputs total points scored for each game
        """
        n = Y.shape[0]
        Score = np.zeros((n,1))
        for i in range(n):
            Score[i] = Y[i,0]+Y[i,1]

        return Score

    def Scramble(self, X_train, Y_train, X_dev, Y_dev, X_test, Y_test):
        """
        Switches winning/losing order of random training examples so the
        algorithm doesn't just think that the first team listed always scores
        more
        """
        half = int(np.shape(X_train)[1]/2)# half of the stats, allows us to
                                          # switch winner and loser stats
        X_data = [X_train, X_dev, X_test]
        Y_data = [Y_train, Y_dev, Y_test]

        for i in range(3):
            for j in range(np.shape(X_data[i])[0]):
                switch = random.random()
                if switch >= 0.5:
                    winscore = Y_data[i][j,0]
                    Y_data[i][j,0] = Y_data[i][j,1]
                    Y_data[i][j,1] = winscore
                    fhalf = X_data[i][j,0:half]
                    lhalf = X_data[i][j,half:]
                    X_data[i][j,:] = np.concatenate((lhalf, fhalf), axis = None)

        return X_data[0], Y_data[0], X_data[1], Y_data[1], X_data[2], Y_data[2]

    def Normalize(self, X_train, X_dev, X_test):
        """
        Normalizes input vectors
        """
        X_data = [X_train, X_dev, X_test]

        for i in range(3):
            mean = np.sum(X_data[i], axis = 0)/np.shape(X_data[i])[0]
            X_data[i] = X_data[i] - mean
            var = np.sum(X_data[i]**2, axis = 0)/np.shape(X_data[i])[0]
            X_data[i] = X_data[i]/np.sqrt(var)

        return X_data[0], X_data[1], X_data[2]

    def Build_Dataset(self, player_path, savepath = "data", Normalized = False):
        """ 
        Builds Train/Dev/Test set for ML Model and saves the data to a .csv

        Args:
            player_path: file containing player box scores in same data structure
            found in the Kaggle dataset at: https://www.kaggle.com/pablote/nba-enhanced-stats
            team_path: file containing team box scores in aforementioned
            structure
            savepath: file name descriptor to fill in the blank: "****_train.csv"
        """
        self.savepath = savepath
        if Normalized:
            Nlabel = "normalized"
        else:
            Nlabel = ""
        #Check if this dataset has been built
        if os.path.isfile('datasets/'+Nlabel+'_train_'+self.savepath+'np'+str(self.num_players)+'ng'+str(self.num_games)+'rs'+\
                str(self.rank_scheme)+'so'+str(len(self.stats_omitted))+'.csv'):
            print("dataset already exists for this configuration")
            return
        #Start by organizing all of the players' statlines {player: {Game Date:
        #statline}}
        player_data = pd.read_csv(player_path)
        players = {}
        for i in range(len(player_data["playLNm"])):
            name = (player_data["playFNm"][i]+" "+player_data["playLNm"][i]).lower()
            if name not in players:
                players[name] = {"GameCount": 1}
            players[name][("GM#"+str(players[name]["GameCount"]), str(player_data["gmDate"][i]))] = \
                [player_data["playMin"][i], player_data["playPTS"][i], player_data["playAST"][i], \
                player_data["playTO"][i], player_data["playSTL"][i], player_data["playBLK"][i], \
                player_data["playPF"][i], player_data["playFGA"][i], player_data["playFGM"][i], \
                player_data["play2PA"][i], player_data["play2PM"][i], player_data["play3PA"][i], \
                player_data["play3PM"][i], player_data["playFTA"][i], player_data["playFTM"][i], \
                player_data["playORB"][i], player_data["playDRB"][i]]
            players[name]["GameCount"] += 1

        #Now do the same for game outcomes {game: {team: {home: ,players: , score: }}}
        games = {}
        for i in range(len(player_data["gmDate"])):
            gamedate = str(player_data["gmDate"][i])
            name = (player_data["playFNm"][i]+" "+player_data["playLNm"][i]).lower()
            matchupf = player_data["teamAbbr"][i]+"vs"+player_data["opptAbbr"][i]
            matchupb = player_data["opptAbbr"][i]+"vs"+player_data["teamAbbr"][i]
            team = player_data["teamAbbr"][i]
            opteam = player_data["opptAbbr"][i]
            if player_data["teamLoc"][i] == "Home":
                home = 1
            else:
                home = 0
            if gamedate+"_"+matchupf not in games and gamedate+"_"+matchupb not in games:
                games[gamedate+"_"+matchupf] = {team: {"home": home, "players": [], "score": 0}, opteam: \
                        {"home": 1-home, "players": [], "score": 0}}
            if gamedate+"_"+matchupf not in games:
                games[gamedate+"_"+matchupb][team]["players"].append(name)
                games[gamedate+"_"+matchupb][team]["score"] += player_data["playPTS"][i]
            else:
                games[gamedate+"_"+matchupf][team]["players"].append(name)
                games[gamedate+"_"+matchupf][team]["score"] += player_data["playPTS"][i]

        #Now build features and labels
        self.X = []
        self.Y = []
        for game in games:
            p = game.find("_")
            gamedate = game[0:p]
            teams = list(games[game].keys())
            if games[game][teams[0]]["home"] == 1:
                teamw = teams[0]
                teaml = teams[1]
            else:
                teamw = teams[1]
                teaml = teams[0]

            #Get statlines for all players from last num_games games for each team
            playstatsw = np.zeros((len(games[game][teamw]["players"]), 17-len(self.stats_omitted), self.num_games))
            playstatsl = np.zeros((len(games[game][teaml]["players"]), 17-len(self.stats_omitted), self.num_games))
            for team in teams:
                ct = 0 #Need to count to make sure that we have enough players on each team
                for player in games[game][team]["players"]:
                    for gameid in players[player]:
                        if gameid[1] == gamedate: #make sure player played in game in question
                            if int(gameid[0][3:]) > self.num_games: #make sure the player has enough previous games
                                for i in range(self.num_games):
                                    prevgame = "GM#"+str(int(gameid[0][3:])-i-1)
                                    for gameidp in players[player]:
                                        if prevgame == gameidp[0]:
                                            if team == teamw:
                                                playstatsw[ct,:,i] = players[player][gameidp]
                                            else:
                                                playstatsl[ct,:,i] = players[player][gameidp]
                    ct += 1
            
            if ct >= self.num_players:
                #Get rank of players by total scoring
                rankplaysw = np.argsort(np.sum(playstatsw, axis = 2)[:,1])[-self.num_players:]
                rankplaysl = np.argsort(np.sum(playstatsl, axis = 2)[:,1])[-self.num_players:]

                #Now let's (finally) build our feature vectors
                x = np.zeros(((17-len(self.stats_omitted))*2*self.num_games*self.num_players,))
                for i in range(self.num_players*2):
                    if i < self.num_players:
                        for j in range(self.num_games):
                            x[(17-len(self.stats_omitted))*(j+self.num_games*i):\
                                    (17-len(self.stats_omitted))*(j+1+self.num_games*i)]=\
                                    playstatsw[rankplaysw[len(rankplaysw)-1-i],:,j]
                    else:
                        for j in range(self.num_games):
                            x[(17-len(self.stats_omitted))*(j+self.num_games*i):\
                                    (17-len(self.stats_omitted))*(j+1+self.num_games*i)]=\
                                    playstatsl[rankplaysl[len(rankplaysl)-1-i+self.num_players],:,j]

                #This if statement just makes sure there are player stats
                #present in x. The first few games make it past the if
                #statements above since the arrays are initialized as zeros.
                if np.dot(x[0:self.num_games*(17-len(self.stats_omitted))],x[0:self.num_games*(17-len(self.stats_omitted))]) != 0\
                and np.dot(x[-self.num_games*(17-len(self.stats_omitted)):],x[-self.num_games*(17-len(self.stats_omitted)):]) != 0:
                    self.X.append(x)
                    self.Y.append(np.array([games[game][teamw]["score"], games[game][teaml]["score"]]))

        #Write train/dev/test sets in 80/10/10 split
        self.X = np.array(self.X)
        self.Y = np.array(self.Y)
        trainsetsz = int(np.shape(self.X)[0]//1.25)
        X_train = self.X[0:trainsetsz,:]
        Y_train = self.Y[0:trainsetsz,:]

        #Train set was first 80% of games. Dev/test sets are drawn randomly
        #from remaining games to ensure that they come from the same distribution
        devsetsz = (np.shape(self.X)[0]-trainsetsz)//2
        testsetsz = np.shape(self.X)[0]-devsetsz-trainsetsz
        X_dev = np.zeros((devsetsz,np.shape(self.X)[1]))
        Y_dev = np.zeros((devsetsz,np.shape(self.Y)[1]))
        X_test = np.zeros((testsetsz,np.shape(self.X)[1]))
        Y_test = np.zeros((testsetsz,np.shape(self.Y)[1]))
        I = np.arange(trainsetsz,np.shape(self.X)[0])
        np.random.shuffle(I)
        for i in range(np.shape(self.X)[0]-trainsetsz):
            if i < devsetsz:
                X_dev[i,:] = self.X[I[i],:]
                Y_dev[i,:] = self.Y[I[i],:]
            else:
                X_test[i-devsetsz,:] = self.X[I[i],:]
                Y_test[i-devsetsz,:] = self.Y[I[i],:]

        #Scramble data to avoid the model always picking team 1 as the winner
        # VERSION NOTE: this was back when teams were ordered by final score, now they are ordered by home/away
        #X_train, Y_train, X_dev, Y_dev, X_test, Y_test = self.Scramble(X_train, Y_train, X_dev, Y_dev, X_test, Y_test)

        #Normalize if specified
        if Normalized:
            X_train, X_dev, X_test = self.Normalize(X_train, X_dev, X_test)

        with open('datasets/'+Nlabel+'_train_'+self.savepath+'np'+str(self.num_players)+'ng'+str(self.num_games)+'rs'+\
                str(self.rank_scheme)+'so'+str(len(self.stats_omitted))+'.csv', mode='w') as train_file:
            train_writer = csv.writer(train_file, delimiter=',')
            for i in range(np.shape(X_train)[0]):
                train_writer.writerow(np.concatenate((Y_train[i,:], X_train[i,:]), axis=None))
        with open('datasets/'+Nlabel+'_dev_'+self.savepath+'np'+str(self.num_players)+'ng'+str(self.num_games)+'rs'+\
                str(self.rank_scheme)+'so'+str(len(self.stats_omitted))+'.csv', mode='w') as dev_file:
            dev_writer = csv.writer(dev_file, delimiter=',')
            for i in range(np.shape(X_dev)[0]):
                dev_writer.writerow(np.concatenate((Y_dev[i,:], X_dev[i,:]), axis=None))
        with open('datasets/'+Nlabel+'_test_'+self.savepath+'np'+str(self.num_players)+'ng'+str(self.num_games)+'rs'+\
                str(self.rank_scheme)+'so'+str(len(self.stats_omitted))+'.csv', mode='w') as test_file:
            test_writer = csv.writer(test_file, delimiter=',')
            for i in range(np.shape(X_test)[0]):
                test_writer.writerow(np.concatenate((Y_test[i,:], X_test[i,:]), axis=None))