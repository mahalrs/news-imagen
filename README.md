# Newsgen - A Multi-Modal Image Generation Model for News Stories

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
