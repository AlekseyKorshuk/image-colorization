from datasets import load_dataset
import os
import uuid

ds = load_dataset("ummagumm-a/colorization_dataset2")
ds = ds["train"].train_test_split(test_size=0.2)

os.makedirs('/code/DATASET/imagenet/train/', exist_ok=True)
os.makedirs('/code/DATASET/imagenet/test/', exist_ok=True)

for img in ds["train"]["image"]:
    name = str(uuid.uuid4())
    img.save(f'/code/DATASET/imagenet/train/{name}.jpeg', 'JPEG')

for img in ds["test"]["image"]:
    name = str(uuid.uuid4())
    img.save(f'/code/DATASET/imagenet/test/{name}.jpeg', 'JPEG')
