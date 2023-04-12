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

import { Browser } from 'puppeteer-core';
import { extractPageContentAndUrls } from './core';
import { CrawlDestination, CrawlInput } from './types';

/**
 * A simple local web crawler which runs in a loop.
 * 
 * @param browser puppeteer browser instance
 * @param input details of the websites to crawl
 * @param destination details of where to write page content
 */
export const crawl = async (
    browser: Browser,
    input: CrawlInput,
    destination: CrawlDestination,
):Promise<{visitedUrls: string[]}> => {
    const urlQueue = [...input.startUrls];
    const seenUrls = new Set(urlQueue);

    while (urlQueue.length > 0) {
        console.log(`\n${urlQueue.length} urls to crawl`);
        
        const url = urlQueue.pop()!;
        seenUrls.add(url);
        console.log('Visiting...', url);

        const newUrls = await extractPageContentAndUrls(browser, {
            ...input,
            url
        }, destination);
        console.log(`    Found ${newUrls.length} urls`);

        urlQueue.push(...newUrls.filter((newUrl) => !seenUrls.has(newUrl)));
    }

    return { visitedUrls: [...seenUrls] };
};
