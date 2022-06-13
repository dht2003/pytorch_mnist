import matplotlib.pyplot as plt

def visualize_data(data,num_visualize):
    fig = plt.figure()
    for i in range(num_visualize):
        plt.subplot(num_visualize // 2, 2, i + 1)
        plt.imshow(data[i][0], cmap='gray', interpolation='none')
        plt.xticks([])
        plt.yticks([])
    return fig