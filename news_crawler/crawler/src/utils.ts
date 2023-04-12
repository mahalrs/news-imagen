// Copyright 2023 The Newsgen Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import * as crypto from 'crypto';
import * as fs from 'fs';
import { CrawlInput } from './types';

/**
 * Create sha1 hash of given url
 * 
 * @param url url to hash
 * @return sha1 hash of the given url
 */
export const getHashHex = (url: string): string => {
    const shasum = crypto.createHash('sha1');
    shasum.update(url);
    return shasum.digest('hex');
};

/**
 * Return crawl input from the given json file
 * 
 * @param path path to config json file
 * @return crawl input
 */
export const createCrawlInput = (path: string): CrawlInput => {
    try {
        const fileData = fs.readFileSync(path, 'utf-8');
        const input: CrawlInput = JSON.parse(fileData);
        return input;
    } catch (err) {
        console.error('Error reading config file', err);
        process.exit(1);
    }
};
