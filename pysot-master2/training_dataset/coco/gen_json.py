from pycocotools.coco import COCO
from os.path import join
import json


dataDir = '.'
dataType = 'Sequences3'

dataset = dict()
annFile = '/media/lyf/工作/study/kd_lowillum/UAVDark135_TSP_out/train_data4.json'
coco = COCO(annFile)
n_imgs = len(coco.imgs)
# for n, img_id in enumerate(coco.imgs):
#     print('subset: {} image id: {:04d} / {:04d}'.format(dataType, n, n_imgs))
#     img = coco.loadImgs(img_id)[0]
#     annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
#     anns = coco.loadAnns(annIds)
#     video_crop_base_path = join(dataType, img['file_name'].split('/')[-1].split('.')[0])


# set_crop_base_path = join(crop_path, dataType)

for video in coco.dataset['videos']:
    video_id = video['id']
    img = []
    anns = []
    for img_single in coco.imgs.values():

        if img_single['video_id'] == video_id:
            img.append(img_single)

    for ann_single in coco.anns.values():
        if ann_single['video_id'] == video_id:
            anns.append(ann_single)

    video_crop_base_path = join(dataType, video['name'])

    if len(anns) > 0:
        dataset[video_crop_base_path] = dict()

    for trackid, ann in enumerate(anns):
        rect = ann['bbox']
        c = ann['category_id']
        bbox = [rect[0], rect[1], rect[0]+rect[2], rect[1]+rect[3]]
        if rect[2] <= 0 or rect[3] <= 0:  # lead nan error in cls.
            continue
        dataset[video_crop_base_path]['{:02d}'.format(trackid)] = {'000000': bbox}

print('save json (dataset), please wait 20 seconds~')
json.dump(dataset, open('train_data.json', 'w'), indent=4)
print('done!')

