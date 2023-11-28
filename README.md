# Image Colorization with Diffusions

## Data preprocessing

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AlekseyKorshuk/image-colorization/blob/main/data_preprocessing.ipynb)

## Training

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AlekseyKorshuk/image-colorization/blob/main/train_image_colorization.ipynb)

# Image Colorization with GANs
build a docker image:

```
docker build . -f docker/Dockerfile.gpu -t image-colorization
```

run a container:

```
docker run -itd --rm -p 7608:7608 --name image-colorization-cont image-colorization
```

run the training:

```
python3.7 download_data.py
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3.7 ChromaGAN.py
```

run the demo:

```
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3.7 app.py
```
