import matplotlib.pyplot as plt

def plot(x, y = None, savepath = "plot.png"):
    if y == None:
        plt.plot(x)
        plt.savefig(savepath)