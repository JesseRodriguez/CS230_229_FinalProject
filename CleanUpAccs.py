import numpy as np
import utilities

SVMOutAccd = np.loadtxt("accuracies/dev_set_acc_SVM.txt")
SVMOutAcct = np.loadtxt("accuracies/train_set_acc_SVM.txt")
LROutAccd = np.loadtxt("accuracies/dev_set_acc_LR.txt")
LROutAcct = np.loadtxt("accuracies/train_set_acc_LR.txt")
QDAOutAccd = np.loadtxt("accuracies/dev_set_acc_QDA.txt")
QDAOutAcct = np.loadtxt("accuracies/train_set_acc_QDA.txt")

np.savetxt("accuracies/dev_set_acc_SVM_clean.txt", SVMOutAccd, fmt="%2.4f")
np.savetxt("accuracies/train_set_acc_SVM_clean.txt", SVMOutAcct, fmt="%2.4f")
np.savetxt("accuracies/dev_set_acc_LR_clean.txt", LROutAccd, fmt="%2.4f")
np.savetxt("accuracies/train_set_acc_LR_clean.txt", LROutAcct, fmt="%2.4f")
np.savetxt("accuracies/dev_set_acc_QDA_clean.txt", QDAOutAccd, fmt="%2.4f")
np.savetxt("accuracies/train_set_acc_QDA_clean.txt", QDAOutAcct, fmt="%2.4f")

utilities.HeatMap(LROutAccd, savepath = "accuracies/LR_dev.pdf")
utilities.HeatMap(LROutAcct, savepath = "accuracies/LR_train.pdf")
utilities.HeatMap(SVMOutAccd, savepath = "accuracies/SVM_dev.pdf")
utilities.HeatMap(SVMOutAcct, savepath = "accuracies/SVM_train.pdf")
utilities.HeatMap(QDAOutAccd, savepath = "accuracies/QDA_dev.pdf")
utilities.HeatMap(QDAOutAcct, savepath = "accuracies/QDA_train.pdf")

NNModel = ["NNF_10L_5x5_f_5x4_2_exp", "NNF_10L_5x5_f_5x4_149_58_sftmx", "NNS_6L_5x5_1_sig"]
for i in range(len(NNModel)):
    NNOutAccd = np.loadtxt("accuracies/dev_set_acc_"+NNModel[i]+".txt")
    NNOutAcct = np.loadtxt("accuracies/train_set_acc_"+NNModel[i]+".txt")
    np.savetxt("accuracies/dev_set_acc_"+NNModel[i]+"_clean.txt", NNOutAccd, fmt="%2.4f")
    np.savetxt("accuracies/train_set_acc_"+NNModel[i]+"_clean.txt", NNOutAcct, fmt="%2.4f")
    utilities.HeatMap(NNOutAccd, savepath = "accuracies/"+NNModel[i]+"_dev.pdf")
    utilities.HeatMap(NNOutAcct, savepath = "accuracies/"+NNModel[i]+"_train.pdf")
