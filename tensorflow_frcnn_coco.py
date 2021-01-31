import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from matplotlib import patches

# load dataset, make it iterable with iter()
ds = iter(tfds.load(name="coco/2017:1.1.0")["validation"])

image = next(ds)["image"]  # take only one image


# load choosen model
model = hub.load("https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_640x640/1")
res = model([image])  # make prediction; input should be collection of images

# get dimensions of the image; we will need it later
im_h, im_w, im_c = image.numpy().shape

# prepare axes for plotting; load image
fig, ax = plt.subplots(1)
ax.imshow(image.numpy())


# from prediction get bounding boxes, predicted category and score for category
for box, cat_id, score in zip(  # iterate over objects in bboxes, categories and scores
        res["detection_boxes"].numpy()[0],
        res["detection_classes"].numpy()[0],
        res['detection_scores'].numpy()[0]
):
    # filter only objects that have probability over 50%
    if score > 0.5:  
        y1, x1, y2, x2 = box  # predicted bounding box format is ymin, xmin, ymax, xmax

        # bbox coordinates are from 0 to 1
        # we need to make them 0 to W for x and 0 to H for y
        y1 = y1 * im_h  # that is why we needed image dimensions
        y2 = y2 * im_h
        x1 = x1 * im_w
        x2 = x2 * im_w
        w, h = x2 - x1, y2 - y1  # calculate width and height for matplotlib
        # draw rectabngle
        rect = patches.Rectangle((x1, y1), w, h, fill=False, edgecolor='r')

        # print category_id over rectangle
        ax.text(x1, y1, cat_id, c="r")
        ax.add_patch(rect)

# show prepared plot
plt.show()
