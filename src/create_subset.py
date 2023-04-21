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
import random
import shutil
import tarfile

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',
                    default='./data/visual_news',
                    help='Directory containing VisualNews dataset')
parser.add_argument('--data_root',
                    default='./data',
                    help='Directory to save subset')
parser.add_argument('--subset_name',
                    default='visual_news_mini',
                    help='Name to use when saving subset')
parser.add_argument('--subset_ratio',
                    default=0.1,
                    type=float,
                    help='Ratio of images to include in the subset')
parser.add_argument('--seed', default=123, type=int, help='Random seed to use')


def create_subset(dataset_dir, data_filename, ratio, out_dir):
    print('Creating subset from', data_filename)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    data_file_path = os.path.join(dataset_dir, data_filename)
    assert os.path.exists(data_file_path), f'{data_file_path} does not exist.'
    assert os.path.isfile(data_file_path), f'{data_file_path} is not a file.'

    with open(data_file_path, 'r') as f:
        data = json.load(f)

    random.shuffle(data['train'])
    random.shuffle(data['val'])
    random.shuffle(data['test'])

    subset = dict()
    subset['train'] = data['train'][:int(ratio * len(data['train']))]
    subset['val'] = data['val'][:int(ratio * len(data['val']))]
    subset['test'] = data['test'][:int(ratio * len(data['test']))]

    def copy_files(subset_split):
        for story in subset_split:
            image_src = os.path.join(dataset_dir, story['image_path'][2:])
            article_src = os.path.join(dataset_dir, story['article_path'][2:])

            image_dest_dir = os.path.join(
                out_dir, os.path.dirname(story['image_path'][2:]))
            article_dest_dir = os.path.join(
                out_dir, os.path.dirname(story['article_path'][2:]))

            if not os.path.exists(image_dest_dir):
                os.makedirs(image_dest_dir)

            if not os.path.exists(article_dest_dir):
                os.makedirs(article_dest_dir)

            image_dest = os.path.join(out_dir, story['image_path'][2:])
            article_dest = os.path.join(out_dir, story['article_path'][2:])

            shutil.copyfile(image_src, image_dest)
            shutil.copyfile(article_src, article_dest)

    print('Copying files...')
    copy_files(subset['train'])
    copy_files(subset['val'])
    copy_files(subset['test'])

    with open(os.path.join(out_dir, data_filename), 'w') as f:
        json.dump(subset, f)


def main():
    args = parser.parse_args()

    # Set the random seed for reproducible experiments
    random.seed(args.seed)

    out_dir = os.path.join(args.data_root, args.subset_name)
    create_subset(args.dataset, 'headlines.json', args.subset_ratio, out_dir)
    create_subset(args.dataset, 'captions.json', args.subset_ratio, out_dir)

    with tarfile.open(f'{out_dir}.tar.gz', 'w:gz') as tar:
        tar.add(out_dir, arcname=args.subset_name)

    # cleanup
    shutil.rmtree(out_dir)


if __name__ == '__main__':
    main()
