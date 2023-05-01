# Newsgen - Multi-Modal Image Generation for News Stories

This repository contains source code for various experiments to explore the feasibility of using a multi-modal approach, inspired by DALL-E mini, to generate images using news headlines and captions. The model is trained on a dataset of news articles with the corresponding images.


## Install dependencies

```sh
pip install -r requirements.txt
```


## Download Data

Newsgen uses VisualNews dataset. It contains a large collection of news stories
from The Guardian, BBC, USA Today, and The Washington Post.
It has over one millon news images along with associated news articles,
image captions, author information, and other metadata. To read more, visit
[VisualNews : A Large Multi-source News Image Dataset](https://www.arxiv-vanity.com/papers/2010.03743/).

We stored a copy of VisualNews dataset in the Google Storage Bucket.
If you have access to it, you can download it as follows:

Requirements:
- Access to storage bucket containing the dataset
- 100 GB free disk space to download. 300 GB recommended as it is needed later for processing.

```sh
# Create a directory to download data
mkdir ./data

# Replace 'my-bucket' with the actual bucket containing the dataset
gsutil cp gs://my-bucket/VisualNews/articles.tar.gz ./data
gsutil cp gs://my-bucket/VisualNews/origin.tar ./data
```

If you don't have access to our storage bucket, visit
[VisualNews-Repository](https://github.com/FuxiaoLiu/VisualNews-Repository) to
find instructions to request and download the dataset.


## Process Data

The downloaded dataset requires some preprocessing before we can use it.

Run the following commands to process the data.
Processing will take about 40 minutes.

Requirements:
- VisualNews dataset: articles.tar.gz and origin.tar files
- 300 GB free disk space
- 8 CPUs with 30 GB RAM

```sh
cd src

# --dataset  : directory containing articles.tar.gz and origin.tar files
# --data_root: directory to save processed data
python process_data.py --dataset ./data --data_root ./data
```


## Run News Crawler

Initially, we planned to scrape news articles to create our own custom dataset.
However, if we encountered many issues, especially deciding which image to use
as the main image as news articles have multiple images including images related
to ads. Given the limited time, we decided to use the VisualNews dataset.

Steps to run the news crawler are as follows:

```sh
cd news_crawler/crawler

npm install
npm run build
npm run start -- --crawl-config ./crawl-input.json --dir-path ./data
```


## Train / Fine-tune VQGAN

Before we train or fine-tune VQGAN, we need to download few pre-trained models.

- Download VQGAN checkpoint pre-trained on ImageNet. See https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/ for more details.
- Download LPIPS pre-trained model. See https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b for more details.

```sh
cd src
mkdir pretrained
# Move downloaded checkpoints in ./pretrained
```
Preprocess the downloaded pre-trained model

``sh
python process_pretrained_vqgan.py
python -m lightning.pytorch.utilities.upgrade_checkpoint ./pretrained/vqgan.ckpt
``

To start training/fine-tuning:

```sh
python trainer_vqgan.py

# OR
# python trainer_vqgan.py --dataset ./data/visual_news_mini --val_every_n_steps 100 --log_every_n_steps 100 --epochs 5
```

NOTE: If you get a `RuntimeError: tensorflow/compiler/xla/xla_client/computation_client.cc:280 : Missing XLA configuration` error on GCP, just do `pip uninstall torch_xla`.


### Visualization

By default, VQGAN trainer stores logs to `logs` directory. To visualize in Tensorboard, run:

```sh
tensorboard --logdir ./logs
```


### Evaluate VQGAN

To evaluate the fine-tuned VQGAN, run the following Jupyter notebook:
`./notebooks/eval_vqgan.ipynb`


## Train BART Decoder

Before we train the decoder, let's encode our dataset to speed up the training.

``sh
python encode_data.py
``

To start training:

```sh
cd src

python trainer_newsgen.py --epochs 5 --warmup_steps 4000 --total_steps 40000
```

### Visualization

By default, the trainer stores logs to `logs` directory. To visualize in Tensorboard, run:

```sh
tensorboard --logdir ./logs
```


### Evaluate the Decoder

To evaluate the decoder, run the following Jupyter notebook:
`./notebooks/eval_decoder.ipynb`
