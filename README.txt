-------------------------------------------------------------------------------
This is the README for Jesse Rodriguez's CS 229/230 Final Projects
-------------------------------------------------------------------------------
Objective: This code creates various machine learning models which are trained
to predict the outcome of NBA basketball games. See final report for further
details.

-------------------------------------------------------------------------------
File Glossary
-------------------------------------------------------------------------------
- nba-enhanced-stats: Folder containing base csv file for dataset

- BuildAll.py: Python script that builds formatted datasets for every num_games
  num_players combo up to a specified limit for each (10x10 currently)

- CleanUpAccs.py: Python script that produces nice accuracy heatmap plots

- LogReg.py: Python script that executes custom logistic regression classifier.

- LogisticRegression.py: Custom Logistic Regression classifier class. Rendered
  irrelevant by later usage of SKLearn

- NN.py: Python script that executes NN builder.

- NNDev.py: Python script that uses the NN builder class to build and test
  various NN models.

- NeuralNetwork.py: Fully connected neural network builder that utilizes the Keras
  framework. An earlier version of the NN builder also exists in this file
  that utilizes an outdated tensorflow framework.

- RNBA.py: Dataset manipulation class that has various helper functions used
  to manipulate training examples and predictions, along with the base dataset
  builder.

- SKLearnModels.py: Runs all classifier models that I chose for this problem
  for each num_players and num_games combo and give accuracy on train and dev
  sets

- TestAll.py: Python script that tests all models with parameters determined 
  during development on the test sets

- utilities.py: Helper functions, mostly for plotting

- Visualize.py: Python script that produces histograms of the datasets