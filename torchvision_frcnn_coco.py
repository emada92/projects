from torchvision.datasets import CocoDetection
from torchvision.transforms.functional import to_tensor, to_pil_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from matplotlib import pyplot as plt, patches


def prepare_image(img):
    return to_tensor(img)


dataset = CocoDetection(
    root="C:/Users/emila/val2017",
    annFile="C:/Users/emila/annotations/instances_val2017.json",
    transform=prepare_image
)

model = fasterrcnn_resnet50_fpn(
    pretrained=True,
    progress=True
)
image = dataset[0][0]
print(dataset[0])


model.eval()
res = model([image])

fig, ax = plt.subplots(1)
ax.imshow(to_pil_image(image))

for box, cat_id in zip(res[0]["boxes"], res[0]["labels"]):
    x1, y1, x2, y2 = box
    w, h = x2-x1, y2-y1
    rect = patches.Rectangle((x1, y1), w, h, fill=False, edgecolor='r')

    category = dataset.coco.loadCats(cat_id.numpy().tolist())[0]["name"]
    ax.text(x1, y1, category, c="r")
    ax.add_patch(rect)

plt.savefig('torchvision_img.png');