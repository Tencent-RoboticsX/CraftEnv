import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns; sns.set()

plt.switch_backend('agg')

info = {
    "QMIX_TEST":{
        "lr=0.005, epsilon=0.2": [15, 17, 16],
        "lr=0.005, epsilon=0.3": [18, 20, 19],
        "lr=0.0005, epsilon=0.2": [23, 21, 22],
        "lr=0.0005, epsilon=0.3": [25, 26, 24],
    }
}

info = {
    "The complex two-story building task":{ 
        "QMIX": [40, 28, 27], # 40 43 28 27
        "QTRAN": [48, 49, 36, 35, ],
        "VDN": [47, 41, 38, 37, ],
        "IQL": [42, 44, 32, 31],
        "MAPPO": [45, 51, 34, 33, ],
        "COMA": [46, 30]
    },
    "The simple two-story building task":{
        "QMIX": [90, 71, 82, 68, ],
        "QTRAN": [84, 77, 81, 67, ],
        "VDN": [80, 75, 87, 76],
        "IQL": [79, 72, 85, 74],
        "MAPPO": [83, 69, 89, 73, ],
        "COMA": [88, 70, 86, 66]
    },
    "The block-shaped building task":{
        "QMIX": [104, 105, 106, 107, 108],
        "QTRAN": [109, 110, 111, 112, 113],
        "VDN": [114, 115, 116, 117, 118],
        "IQL": [119, 120, 121, 122, 123],
        "MAPPO": [124, 125, 126, 127, 128],
        "COMA": [130, 131, 132] # 129 133
    },
    "The strip-shaped building task": {
        "QMIX": [134, 135, 136, 137, 138],
        "QTRAN": [139, 140, 141, 142, 143],
        "VDN": [144, 145, 146, 147, 148],
        "IQL": [149, 150, 151, 152, 153],
        "MAPPO": [154, 155, 156, 157, 158],
        "COMA": [159, 160, 161, 162, 163]
    }
}

data = {}

# read data...
for task, algos in info.items():
    for algo, labels in algos.items():
        for label in labels:
            path = "craft/PyMARL/results/sacred/" + str(label) + "/info.json"
            print(path)
            data1 = json.load(open(path))
            return_mean_T = data1['return_mean_T']
            return_mean = data1['return_mean']
            # print(return_mean_T)
            # print(return_mean)
            test_return_mean_T = data1['test_return_mean_T']
            test_return_mean = data1['test_return_mean']
            # print(test_return_mean_T)
            # print(test_return_mean)

            x = np.array(return_mean_T)
            y = np.array(return_mean)

            x = np.array(test_return_mean_T)
            y = np.array(test_return_mean)

            if task not in data:
                data[task] = {}
            if algo not in data[task]:
                data[task][algo] = []
            data[task][algo].append((x, y))


# plot!
from matplotlib.pyplot import figure
figure(figsize=(5, 4), dpi=80)
for task in sorted(data.keys()):
    plt.clf()
    for algo in sorted(data[task].keys()):
        xs, ys = zip(*data[task][algo])
        xs, ys = np.array(xs), np.array(ys)
        def cut(x, length):
            x_cut = np.empty([x.shape[0], length])
            for i in range(x.shape[0]):
                x_cut[i] = x[i][0:length]
            return x_cut
        min_length = min(xs[i].shape[0] for i in range(xs.shape[0]))
        if task in ["task 06", "task 07"]:
            cutter = -1
            for i in range(0, xs[0].shape[0]):
                if xs[0][i] >= 1e6:
                    cutter = i
                    break
            min_length = min(min_length, cutter)
        print(min_length)
        xs = cut(xs, min_length)
        ys = cut(ys, min_length)
        assert xs.shape == ys.shape
        label = algo
        # Calculate for success rate
        success_rate_flag = False
        if success_rate_flag:
            success_rate = [[] for _ in range(0, len(ys))]
            max_val = None
            if task == "The strip-shaped building task":
                ranges = [8e5, 1e6]
                max_val = 4
            if task == "The block-shaped building task":
                ranges = [8e5, 1e6]
                max_val = 4
            if task == "The simple two-story building task":
                ranges = [1.5e6, 2e6]
                max_val = 3
            if task == "The complex two-story building task":
                ranges = [1.5e6, 2e6]
                max_val = 20
            if max_val is None:
                continue
            else:
                for i in range(0, len(ys[0])):
                    for j in range(0, len(ys)):
                        success = 0
                        total = 1
                        if ys[j][i] >= max_val:
                            success += 1
                        success_rate[j].append(success / total)
            print(len(success_rate))
            ys = np.array(success_rate)
            print(ys.shape)

        plt.plot(xs[0], np.mean(ys, axis=0), label=label, linewidth=2, alpha=1.)
        plt.fill_between(xs[0], np.mean(ys, axis=0)+np.std(ys, axis=0), np.mean(ys, axis=0)-np.std(ys, axis=0), alpha=0.25)
    plt.title('{}'.format(task))
    plt.legend()
    if success_rate_flag:
        plt.savefig("SR_{}.pdf".format(task), bbox_inches='tight')
    else:
        plt.savefig("{}.pdf".format(task), bbox_inches='tight')

        
