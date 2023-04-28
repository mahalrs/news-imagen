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

    def __init__(self, encoded_dataset, split):
        self.data = self.get_data(encoded_dataset, split)
        print('Data size:', len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item[0], item[1], item[2]

    def get_data(self, encoded_dataset, split):
        assert split in ['train', 'val', 'test'], f'Invalid split: {split}'
        BOS_TOKEN = 16385
        data = []
        with open(encoded_dataset, 'r') as f:
            for story in json.load(f)[split]:
                item = []
                item.append(story['input_tokens'])
                item.append(story['attention_mask'])
                item.append([BOS_TOKEN] + story['labels'])

                data.append(item)

        return data
