# Copyright 2023 The Newsgen Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os
import pickle
import shutil
import tarfile

parser = argparse.ArgumentParser()
parser.add_argument(
    '--dataset',
    default='./data',
    help=
    'Directory containing VisualNews dataset: articles.tar.gz and origin.tar files'
)
parser.add_argument('--dataset_name',
                    default='visual_news',
                    help='Directory name to use when saving processed dataset')
parser.add_argument('--data_root',
                    default='./data',
                    help='Directory to save processed dataset')


def make_output_dir(out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)


def untar(tar_file, out_dir, out_name=None, gzip=True):
    mode = 'r:gz' if gzip else 'r'
    with tarfile.open(tar_file, mode) as tar:
        tar.extractall(path=out_dir)

        if out_name:
            if os.path.exists(os.path.join(out_dir, out_name)):
                shutil.rmtree(os.path.join(out_dir, out_name))

            os.rename(os.path.join(out_dir,
                                   tar.getnames()[0]),
                      os.path.join(out_dir, out_name))


def process_bbc1(articles_dir):
    raw_data_file = os.path.join(articles_dir, 'bbc_1.tar.gz')
    processed_data_file = os.path.join(articles_dir, 'processed_bbc_1.p')

    out_name = 'bbc_1'
    out_dir = os.path.join(articles_dir, out_name)
    file_prefix = 'http:::walter-producer-cdn.api.bbci.co.uk:content'

    headlines = dict()

    untar(raw_data_file, articles_dir, out_name=out_name)

    with open(processed_data_file, 'rb') as f:
        data = pickle.load(f)

    for k, v in data.items():
        filename = file_prefix + v['article_id'] + '-200.json'
        filename = filename.replace('/', ':')
        article_path = os.path.join(out_dir, filename)
        with open(article_path) as f:
            article = json.load(f)
            headlines[k] = {'headline': article['name']}

    # cleanup
    shutil.rmtree(os.path.join(articles_dir, out_name))

    return headlines


def process_usa_today(articles_dir):
    raw_data_file = os.path.join(articles_dir, 'usa_today_articles.tar.gz')
    processed_data_file = os.path.join(articles_dir, 'processed_usa_today.p')

    out_name = 'usa_today'
    out_dir = os.path.join(articles_dir, out_name)

    headlines = dict()

    untar(raw_data_file, articles_dir, out_name=out_name, gzip=False)

    with open(processed_data_file, 'rb') as f:
        data = pickle.load(f)

    for k, v in data.items():
        article_path = os.path.join(out_dir, f"{v['article_id']}.json")
        with open(article_path) as f:
            article = json.load(f)
            headlines[k] = {'headline': article['headline']}

    # cleanup
    shutil.rmtree(os.path.join(articles_dir, out_name))

    return headlines


def process_data(dataset_dir, articles_dir):
    headlines = dict()

    headlines.update(process_bbc1(articles_dir))
    headlines.update(process_usa_today(articles_dir))

    headlines_data = dict()
    captions_data = dict()

    with open(os.path.join(dataset_dir, 'data.json'), 'r') as f:
        data = json.load(f)

    for item in data:
        idx = item['id']
        if idx in headlines:
            headlines_data[idx] = item
            headlines_data[idx]['headline'] = headlines[idx]['headline']
        else:
            captions_data[idx] = item

    assert len(headlines_data) + len(captions_data) == len(
        data), 'len headlines and captions dict does not match len data'

    print('Headlines:', len(headlines_data))
    print('Captions:', len(captions_data))

    headlines_to_delete = []
    for k, v in headlines_data.items():
        if not os.path.exists(os.path.join(
                dataset_dir, v['image_path'][2:])) or not os.path.exists(
                    os.path.join(dataset_dir, v['article_path'][2:])):
            headlines_to_delete.append(k)

    for k in headlines_to_delete:
        del headlines_data[k]

    captions_to_delete = []
    for k, v in captions_data.items():
        if not os.path.exists(os.path.join(
                dataset_dir, v['image_path'][2:])) or not os.path.exists(
                    os.path.join(dataset_dir, v['article_path'][2:])):
            captions_to_delete.append(k)

    for k in captions_to_delete:
        del captions_data[k]

    print('Headlines:', len(headlines_data))
    print('Captions:', len(captions_data))

    with open(os.path.join(dataset_dir, 'headlines.json'), 'w') as f:
        json.dump(list(headlines_data.values()), f)

    with open(os.path.join(dataset_dir, 'captions.json'), 'w') as f:
        json.dump(list(captions_data.values()), f)

    # cleanup
    os.remove(os.path.join(dataset_dir, 'data.json'))


def main():
    args = parser.parse_args()

    origin_tar = os.path.join(args.dataset, 'origin.tar')
    assert os.path.exists(origin_tar), f'{origin_tar} does not exist.'
    assert os.path.isfile(origin_tar), f'{origin_tar} is not a file.'

    articles_tar = os.path.join(args.dataset, 'articles.tar.gz')
    assert os.path.exists(articles_tar), f'{articles_tar} does not exist.'
    assert os.path.isfile(articles_tar), f'{articles_tar} is not a file.'

    make_output_dir(args.data_root)

    print('Extracting files...')
    untar(origin_tar, args.data_root, out_name=args.dataset_name, gzip=False)
    untar(articles_tar, args.data_root)

    dataset_dir = os.path.join(args.data_root, args.dataset_name)
    articles_dir = os.path.join(args.data_root, 'articles')
    print('Processing...')
    process_data(dataset_dir, articles_dir)

    # cleanup
    shutil.rmtree(os.path.join(args.dataset, 'articles'))


if __name__ == '__main__':
    main()
