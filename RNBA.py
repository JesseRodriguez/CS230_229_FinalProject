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
            train_data = pd.read_csv('datasets/binary_train_'+self.savepath+'np'+str(self.num_players)+'ng'+str(self.num_games)+'rs'+\
                    str(self.rank_scheme)+'so'+str(len(self.stats_omitted))+'.csv',header = None).to_numpy()
            test_data = pd.read_csv('datasets/binary_test_'+self.savepath+'np'+str(self.num_players)+'ng'+str(self.num_games)+'rs'+\
                    str(self.rank_scheme)+'so'+str(len(self.stats_omitted))+'.csv',header = None).to_numpy()
            dev_data = pd.read_csv('datasets/binary_dev_'+self.savepath+'np'+str(self.num_players)+'ng'+str(self.num_games)+'rs'+\
                    str(self.rank_scheme)+'so'+str(len(self.stats_omitted))+'.csv',header = None).to_numpy()
            X_train = train_data[:,1:]
            Y_train = train_data[:,0]
            X_test = test_data[:,1:]
            Y_test = test_data[:,0]
            X_dev = dev_data[:,1:]
            Y_dev = dev_data[:,0]
        else:
            train_data = pd.read_csv('datasets/train_'+self.savepath+'np'+str(self.num_players)+'ng'+str(self.num_games)+'rs'+\
                    str(self.rank_scheme)+'so'+str(len(self.stats_omitted))+'.csv',header = None).to_numpy()
            test_data = pd.read_csv('datasets/test_'+self.savepath+'np'+str(self.num_players)+'ng'+str(self.num_games)+'rs'+\
                    str(self.rank_scheme)+'so'+str(len(self.stats_omitted))+'.csv',header = None).to_numpy()
            dev_data = pd.read_csv('datasets/dev_'+self.savepath+'np'+str(self.num_players)+'ng'+str(self.num_games)+'rs'+\
                    str(self.rank_scheme)+'so'+str(len(self.stats_omitted))+'.csv',header = None).to_numpy()
            X_train = train_data[:,2:]
            Y_train = train_data[:,0:2]
            X_test = test_data[:,2:]
            Y_test = test_data[:,0:2]
            X_dev = dev_data[:,2:]
            Y_dev = dev_data[:,0:2]
        
        return X_train, Y_train, X_dev, Y_dev, X_test, Y_test

    def Binary_Clf_Dataset(self):
        """
        Builds Train/Dev/Test sets for Binary classifier model
        """
        #Check if this dataset has been built
        if os.path.isfile('datasets/binary_train_'+self.savepath+'np'+str(self.num_players)+'ng'+str(self.num_games)+'rs'+\
                str(self.rank_scheme)+'so'+str(len(self.stats_omitted))+'.csv'):
            print("binary classifier dataset already exists for this configuration")
            return

        X_train, Y_train, X_dev, Y_dev, X_test, Y_test = self.Read_Dataset()
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

        with open('datasets/binary_train_'+self.savepath+'np'+str(self.num_players)+'ng'+str(self.num_games)+'rs'+\
                str(self.rank_scheme)+'so'+str(len(self.stats_omitted))+'.csv', mode='w') as train_file:
            train_writer = csv.writer(train_file, delimiter=',')
            for i in range(np.shape(X_train)[0]):
                train_writer.writerow(np.concatenate((Y_train_b[i,0], X_train[i,:]), axis=None))
        with open('datasets/binary_dev_'+self.savepath+'np'+str(self.num_players)+'ng'+str(self.num_games)+'rs'+\
                str(self.rank_scheme)+'so'+str(len(self.stats_omitted))+'.csv', mode='w') as dev_file:
            dev_writer = csv.writer(dev_file, delimiter=',')
            for i in range(np.shape(X_dev)[0]):
                dev_writer.writerow(np.concatenate((Y_dev_b[i,0], X_dev[i,:]), axis=None))
        with open('datasets/binary_test_'+self.savepath+'np'+str(self.num_players)+'ng'+str(self.num_games)+'rs'+\
                str(self.rank_scheme)+'so'+str(len(self.stats_omitted))+'.csv', mode='w') as test_file:
            test_writer = csv.writer(test_file, delimiter=',')
            for i in range(np.shape(X_test)[0]):
                test_writer.writerow(np.concatenate((Y_test_b[i,0], X_test[i,:]), axis=None))

    def Scramble(self):
        """
        Switches winning/losing order of random training examples so the
        algorithm doesn't just think that the first team listed always scores
        more
        """
        X_train, Y_train, X_dev, Y_dev, X_test, Y_test = self.Read_Dataset()
        half = int(np.shape(X_train)[1]/2)# half of the stats, allows us to
                                          # switch winner and loser stats

        for i in range(np.shape(X_train)[0]):
            switch = random.random()
            if switch >= 0.5:
                winscore = Y_train[i,0]
                Y_train[i,0] = Y_train[i,1]
                Y_train[i,1] = winscore
                fhalf = X_train[i,0:half]
                lhalf = X_train[i,half:]
                X_train[i,:] = np.concatenate((lhalf, fhalf), axis = None)
        for i in range(np.shape(X_dev)[0]):
            switch = random.random()
            if switch >= 0.5:
                winscore = Y_dev[i,0]
                Y_dev[i,0] = Y_dev[i,1]
                Y_dev[i,1] = winscore
                fhalf = X_dev[i,0:half]
                lhalf = X_dev[i,half:]
                X_dev[i,:] = np.concatenate((lhalf, fhalf), axis = None)
        for i in range(np.shape(X_test)[0]):
            switch = random.random()
            if switch >= 0.5:
                winscore = Y_test[i,0]
                Y_test[i,0] = Y_test[i,1]
                Y_test[i,1] = winscore
                fhalf = X_test[i,0:half]
                lhalf = X_test[i,half:]

        with open('datasets/train_'+self.savepath+'np'+str(self.num_players)+'ng'+str(self.num_games)+'rs'+\
                str(self.rank_scheme)+'so'+str(len(self.stats_omitted))+'.csv', mode='w') as train_file:
            train_writer = csv.writer(train_file, delimiter=',')
            for i in range(np.shape(X_train)[0]):
                train_writer.writerow(np.concatenate((Y_train[i,:], X_train[i,:]), axis=None))
        with open('datasets/dev_'+self.savepath+'np'+str(self.num_players)+'ng'+str(self.num_games)+'rs'+\
                str(self.rank_scheme)+'so'+str(len(self.stats_omitted))+'.csv', mode='w') as dev_file:
            dev_writer = csv.writer(dev_file, delimiter=',')
            for i in range(np.shape(X_dev)[0]):
                dev_writer.writerow(np.concatenate((Y_dev[i,:], X_dev[i,:]), axis=None))
        with open('datasets/test_'+self.savepath+'np'+str(self.num_players)+'ng'+str(self.num_games)+'rs'+\
                str(self.rank_scheme)+'so'+str(len(self.stats_omitted))+'.csv', mode='w') as test_file:
            test_writer = csv.writer(test_file, delimiter=',')
            for i in range(np.shape(X_test)[0]):
                test_writer.writerow(np.concatenate((Y_test[i,:], X_test[i,:]), axis=None))

    def Normalize(self):
        """
        Switches winning/losing order of random training examples so the
        algorithm doesn't just think that the first team listed always scores
        more
        """
        X_train, Y_train, X_dev, Y_dev, X_test, Y_test = self.Read_Dataset()

        X_data = [X_train, X_dev, X_test]
        Y_data = [Y_train, Y_dev, Y_test]
        label = ["train", "dev", "test"]

        for i in range(3):
            mean = np.sum(X_data[i], axis = 0)/np.shape(X_data[i])[0]
            X_data[i] = X_data[i] - mean
            var = np.sum(X_data[i]**2, axis = 0)/np.shape(X_data[i])[0]
            X_data[i] = X_data[i]/var
            with open('datasets/'+label[i]+'_normalized_'+self.savepath+'np'+str(self.num_players)+'ng'+str(self.num_games)+'rs'+\
                    str(self.rank_scheme)+'so'+str(len(self.stats_omitted))+'.csv', mode='w') as dfile:
                writer = csv.writer(dfile, delimiter=',')
                for j in range(np.shape(X_data[i])[0]):
                    writer.writerow(np.concatenate((Y_data[i][j,:], X_data[i][j,:]), axis=None))

    def Build_Dataset(self, player_path, savepath = "data"):
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
        #Check if this dataset has been built
        if os.path.isfile('datasets/train_'+self.savepath+'np'+str(self.num_players)+'ng'+str(self.num_games)+'rs'+\
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

        #Now do the same for game outcomes {game: {team: {players: , score: }}}
        games = {}
        for i in range(len(player_data["gmDate"])):
            gamedate = str(player_data["gmDate"][i])
            name = (player_data["playFNm"][i]+" "+player_data["playLNm"][i]).lower()
            matchupf = player_data["teamAbbr"][i]+"vs"+player_data["opptAbbr"][i]
            matchupb = player_data["opptAbbr"][i]+"vs"+player_data["teamAbbr"][i]
            team = player_data["teamAbbr"][i]
            opteam = player_data["opptAbbr"][i]
            if gamedate+"_"+matchupf not in games and gamedate+"_"+matchupb not in games:
                games[gamedate+"_"+matchupf] = {team: {"players": [], "score": 0}, opteam: \
                        {"players": [], "score": 0}}
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
            if games[game][teams[0]]["score"] > games[game][teams[1]]["score"]:
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
        with open('datasets/train_'+savepath+'np'+str(self.num_players)+'ng'+str(self.num_games)+'rs'+\
                str(self.rank_scheme)+'so'+str(len(self.stats_omitted))+'.csv', mode='w') as train_file:
            train_writer = csv.writer(train_file, delimiter=',')
            for i in range(trainsetsz):
                train_writer.writerow(np.concatenate((self.Y[i,:], self.X[i,:]), axis=None))

        #Train set was first 80% of games. Dev/test sets are drawn randomly
        #from remaining games to ensure that they come from the same distribution
        I = np.arange(trainsetsz,np.shape(self.X)[0])
        np.random.shuffle(I)
        print(I)
        with open('datasets/test_'+savepath+'np'+str(self.num_players)+'ng'+str(self.num_games)+'rs'+\
                str(self.rank_scheme)+'so'+str(len(self.stats_omitted))+'.csv', mode = 'w') as test_file:
            with open('datasets/dev_'+savepath+'np'+str(self.num_players)+'ng'+str(self.num_games)+'rs'+\
                    str(self.rank_scheme)+'so'+str(len(self.stats_omitted))+'.csv', mode = 'w') as dev_file:
                test_writer = csv.writer(test_file, delimiter=',')
                dev_writer = csv.writer(dev_file, delimiter=',')
                for i in range(np.shape(self.X)[0]-trainsetsz):
                    if i < (np.shape(self.X)[0]-trainsetsz)//2:
                        test_writer.writerow(np.concatenate((self.Y[I[i],:], self.X[I[i],:]), axis=None))
                    else:
                        dev_writer.writerow(np.concatenate((self.Y[I[i],:], self.X[I[i],:]), axis=None))

        self.Scramble()
        self.Normalize()