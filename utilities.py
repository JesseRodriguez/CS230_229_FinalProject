import matplotlib.pyplot as plt
import seaborn as sb

def plot(x, y = None, savepath = "plot.png"):
    if y == None:
        plt.plot(x)
        plt.savefig(savepath)

def histogram(data, xLabel, Title, savepath = "histogram.pdf"):
    plt.hist(data, bins=20)
    plt.tick_params(labelsize='large')
    plt.ylabel('# of games', fontsize='x-large')
    plt.title(Title, fontsize='x-large')
    plt.xlabel(xLabel, fontsize='x-large')
    plt.savefig(savepath)
    plt.close()

def stackedhist(data1, data2, data3, xLabel, Legend, savepath = "histogram.pdf"):
    plt.hist(data1, normed = True, bins=20)
    plt.hist(data2, normed = True, alpha=0.7, bins=20)
    plt.hist(data3, normed = True, alpha=0.7, bins=20)
    plt.tick_params(labelsize='large')
    plt.ylabel('Relative Occurrence', fontsize='x-large')
    plt.title("Comparison", fontsize='x-large')
    plt.xlabel(xLabel, fontsize='x-large')
    plt.legend(Legend, fontsize='x-large')
    plt.savefig(savepath)
    plt.close()

def HeatMap(data, savepath = "heatmap.pdf"):
    heat_map = sb.heatmap(data, annot=True, cbar=False, cmap="RdBu")
    plt.ylabel('Numper of Players - 1')
    plt.xlabel('Number of Games - 1')
    fig = heat_map.get_figure()
    fig.savefig(savepath)
    plt.close()