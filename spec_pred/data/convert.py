# 压缩
import pickle
import json
import numpy as np

datas = []
all_time = []
json_path = "data.json"
train_json_path = "train.json"
dev_json_path = "dev.json"
test_json_path = "test.json"
with open(json_path) as f:
    for line in f:
        item = json.loads(line)
        data = item["data"]
        data = list(data)
        datas.append(data)
        all_time.append(item["date"])
datas = np.asanyarray(datas, np.short)
pickle.dump({"data":datas, "date": all_time}, open("data.pkl", "wb"))

datas2 = pickle.load(open("data.pkl", "rb"))
print(datas2.keys())