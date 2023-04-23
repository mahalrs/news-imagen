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
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from vqgan.model import VQModel

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',
                    default='data/visual_news_mini',
                    help='Directory containing VisualNews dataset')


with open('../src/hparams_vqgan.json', 'r') as f:
        hparams = json.load(f)
model = VQModel.load_from_checkpoint('../src/pretrained/vqgan.ckpt', **hparams)
model.init_lpips_from_pretrained('../src/pretrained/vgg.pth')

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def encode(dataset_dir, file_name):
    with open(os.path.join(dataset_dir, file_name), 'r') as f:
        data = json.load(f)
        for split in ['train', 'val', 'test']:
            for i in range(len(data[split])):
            # for story in data[split]:
                story = data[split][i]
                image_path = os.path.join(dataset_dir, story['image_path'][2:])
                img = Image.open(image_path).convert('RGB')
                transformed = transform(img)
                img_tensor = transform(img).unsqueeze(0)
                output, loss, info = model.encode(img_tensor)
                
                np_encoding = output.detach().numpy()
                story['encoding'] = np_encoding.tolist()
                
        with open(os.path.join(dataset_dir, f"{file_name[:-5]}_encoding.json"), 'w') as f:
            json.dump(data, f)

def main():
    args = parser.parse_args()
    dataset_dir = args.dataset

    torch.set_grad_enabled(False)
    model.eval()
    
    encode(dataset_dir, 'headlines.json')
    print("headlines.json encoded")
    encode(dataset_dir, 'captions.json')
    print("captions.json encoded")

if __name__ == '__main__':
    main()