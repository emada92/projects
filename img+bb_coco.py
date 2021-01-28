import json 
import numpy as np
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
from matplotlib import pyplot as plt, patches
import random
import tensorflow as tf

def save_random_coco_images(num_of_images, category):
    for i in range(num_of_images):
        dataType='val2017'
        annFile='C:/Users/emila/annotations/instances_{}.json'.format(dataType)
        coco=COCO(annFile)
        catIds = coco.getCatIds(catNms=[category])
        img_id_list = coco.getImgIds(catIds = catIds )
        img_id = int(np.random.choice(img_id_list))
        img = coco.loadImgs(img_id)[0]
        
        #draw bouding boxes
        annIds = coco.getAnnIds(imgIds=img['id'])
        anns = coco.loadAnns(annIds)[0]
        bbox = anns['bbox']
        
        x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]

        I = io.imread(img['coco_url'])
        
        fig, ax = plt.subplots(1)
        rect = patches.Rectangle((x, y), w, h, fill=False, edgecolor='r')
        ax.imshow(I)
        ax.add_patch(rect)
        plt.savefig(f'mygraph_{category}{i}.png')
