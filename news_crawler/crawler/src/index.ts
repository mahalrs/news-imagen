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

import puppeteer from 'puppeteer';
import { Command } from 'commander';
import { Browser } from 'puppeteer-core';
import { crawl } from './local';
import { createCrawlInput } from './utils';

const program = new Command();

program
    .option('--crawl-config <path>', 'Path to json file to create crawl input')
    .option('--dir-path <path>', 'Path to directory to store crawled content');

const options = program.parse(process.argv).opts();

(async () => {
    const browser = await puppeteer.launch() as unknown as Browser;
    
    const result = await crawl(browser, createCrawlInput(options.crawlConfig), {
        dirPath: options.dirPath
    });

    await browser.close();

    console.log(result);
})();
