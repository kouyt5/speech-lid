import json
import numpy as np


def plot_graph(data, name):
    import numpy as np
    import matplotlib.pyplot as plt
    import librosa.display
    plt.rcParams["font.sans-serif"]=["STSong"] #设置字体
    plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题

    sr = data.shape[1] * 10 * 2
    D = data.T  # win_length : int <= n_fft [scalar]
    mean = np.mean(D)
    min_value = np.min(D)
    D_denoise = np.zeros_like(D)
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            if D[i][j] >= mean + 80:
                D_denoise[i][j] = D[i][j]
            else:
                D_denoise[i][j] = min_value
    D = D_denoise
    plt.figure(figsize=(20, 10))
    plt.subplot(1,1,1)
    librosa.display.specshow(D,sr=sr,x_axis="s", y_axis="hz", hop_length=1)
    plt.ylabel("Time(s)")
    plt.xlabel("Freq")
    plt.colorbar()
    plt.savefig(f"/hdd/1/chenc/lid/speech-lid/spec_pred/data/img/{name}.png", dpi=300)
    

interval = 100
json_path = "/hdd/1/chenc/lid/speech-lid/spec_pred/data/data.json"
count = 0
with open(json_path) as f:
    
    datas = []
    for line in f:
        item = json.loads(line)
        data = item["data"]
        data = list(data)
        datas.append(data)
        if count < 3300:
            count += 1
            continue
        if count % interval == interval - 1:
            datas = np.asanyarray(datas)
            print(f"count: {count}")
            plot_graph(datas.T, str(count))
            del datas
            datas = []
            # break
        count += 1
            

