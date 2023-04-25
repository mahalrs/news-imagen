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

import json
import os

import torch
from torch.utils.data import Dataset
from PIL import Image


class VQVisualNewsDataset(Dataset):

    def __init__(self, visual_news_dataset_dir, split, transform):
        self.image_paths = self.get_image_paths(visual_news_dataset_dir, split)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        sample = self.image_paths[idx]

        try:
            img = Image.open(sample).convert('RGB')
            return self.transform(img)
        except Exception as exc:
            print(exc)
            return None

    def get_image_paths(self, dataset_dir, split):
        assert split in ['train', 'val', 'test'], f'Invalid split: {split}'

        img_paths = []

        with open(os.path.join(dataset_dir, 'headlines.json'), 'r') as f:
            for story in json.load(f)[split]:
                img_paths.append(
                    os.path.join(dataset_dir, story['image_path'][2:]))

        with open(os.path.join(dataset_dir, 'captions.json'), 'r') as f:
            for story in json.load(f)[split]:
                img_paths.append(
                    os.path.join(dataset_dir, story['image_path'][2:]))

        return img_paths


class EncodedVisualNewsDataset(Dataset):

    def __init__(self,
                 visual_news_dataset_dir,
                 encoded_data_file,
                 split,
                 headlines=False):
        self.data = self.get_data(visual_news_dataset_dir, encoded_data_file,
                                  split, headlines)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item[0], item[1], item[2]

    def get_data(self, dataset_dir, encoded_data_file, split, headlines=False):
        assert split in ['train', 'val', 'test'], f'Invalid split: {split}'

        data = []

        with open(os.path.join(dataset_dir, encoded_data_file), 'r') as f:
            for story in json.load(f)[split]:
                item = []
                if headlines:
                    item.append(torch.tensor(story['headline_tokens']))
                    item.append(torch.tensor(story['headline_attention']))
                else:
                    item.append(torch.tensor(story['caption_tokens']))
                    item.append(torch.tensor(story['caption_attention']))

                img_tokens = [0]
                img_tokens.extend(story['image_tokens'])
                img_tokens.append(2)
                item.append(torch.tensor(img_tokens))

                data.append(item)

        return data
