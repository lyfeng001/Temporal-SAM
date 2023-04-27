import json
import os

# current_path = os.getcwd()
##
##
path = '/media/lyf/工作/study/kd_lowillum/pysot-master2/training_dataset/coco/train_data.json'
with open(path, "r") as f:
    data = json.load(f)

for id1, no in enumerate(data):
    piss = {}
    new = {}
    for id, pis in enumerate(data[no]):
        key = '{:02d}'.format(id)
        key_new = '{:06d}'.format(int(pis))
        piss[key_new] = data[no][pis]['000000']
    new['{:02d}'.format(int(0))] = piss
    data[no] = new

json_str = json.dumps(data, indent=4)
with open("/media/lyf/工作/study/kd_lowillum/pysot-master2/training_dataset/coco/train_data_new.json", "w") as f:
    f.write(json_str)