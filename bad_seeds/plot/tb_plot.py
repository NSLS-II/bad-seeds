import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import tensorflow as tf
import matplotlib as mpl
mpl.use("Qt5Agg")
import matplotlib.pyplot as plt


def plot_tensorflow_log(path):

    # Loading too much data is slow...
    tf_size_guidance = {
        'compressedHistograms': 10,
        'images': 0,
        'scalars': 100,
        'histograms': 1
    }

    event_acc = EventAccumulator(path, tf_size_guidance)
    event_acc.Reload()

    # Show all tags in the log file
    print(event_acc.Tags())

    timestep_reward = event_acc.Tensors('agent.observe/timestep-reward')
    episode_reward = event_acc.Tensors('agent.observe/episode-reward')

    #print(type(timestep_reward[0]))
    #tensor_event = timestep_reward[0]
    #tensor_np = tf.make_ndarray(tensor_event.tensor_proto)
    #print(tensor_np)

    steps = len(timestep_reward)
    print(f"steps: {steps}")
    x = np.arange(steps)
    y = np.zeros([steps, 2])

    for i in range(steps):
        y[i, 0] = tf.make_ndarray(timestep_reward[i].tensor_proto)
        y[i, 1] = tf.make_ndarray(episode_reward[i].tensor_proto)

    plt.plot(x, y[:,0], label='training accuracy')
    plt.plot(x, y[:,1], label='validation accuracy')

    plt.xlabel("Steps")
    plt.ylabel("Accuracy")
    plt.title("Training Progress")
    plt.legend(loc='upper right', frameon=True)
    plt.show()


if __name__ == '__main__':
    #log_file = "./logs/events.out.tfevents.1456909092.DTA16004"
    log_file = "./training_data/agent_random_env_04/summaries/summary-20200821-004719/events.out.tfevents.1597985240.thorisdottir.140924.11.v2"
    plot_tensorflow_log(log_file)