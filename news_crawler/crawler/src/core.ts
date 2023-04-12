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

import * as fs from 'fs';
import * as path from 'path';
import { Browser, Page } from 'puppeteer-core';
import { URL } from 'url';
import { CrawlDestination, CrawlPageInput, PageContent } from './types';
import { getHashHex } from './utils';

/**
 * Extract content from a page.
 * 
 * @param page puppeteer browser page
 * @return extracted page content
 */
const extractContent = async (page: Page): Promise<PageContent> => {
    const [ title, htmlContent ] = await Promise.all([
        page.evaluate(() => document.title),
        page.evaluate(() => document.body.innerHTML),
    ]);
    return { title, htmlContent };
};

/**
 * Write the given page content to the given destination, along with metadata
 * 
 * @param url page url
 * @param content extracted page content
 * @param pageID unique page id
 * @param destination the location to write content to
 */
const writePageContent = async (
    url: string,
    content: PageContent,
    pageID: string,
    destination: CrawlDestination,
) => {
    if (!content.title || !content.htmlContent) {
        console.log('Page has no content, skipping');
        return;
    }

    try {
        if (!fs.existsSync(destination.dirPath)) {
            console.log('Creating destination dir');
            fs.mkdirSync(destination.dirPath, { recursive: true });
        }

        const documentKey = path.join(destination.dirPath, pageID);
        fs.writeFileSync(`${documentKey}.html`, content.htmlContent);

        const metadata = {
            Title: content.title,
            SourceURL: url,
            PageID: pageID,
        }

        fs.writeFileSync(`${documentKey}.metadata.json`, JSON.stringify(metadata));
    } catch (err) {
        console.error(err);
    }
};

/**
 * Return all the urls from a page
 * 
 * @param page puppeteer browser page
 * @param targetWebsites key is the target domain and value is the list of
 *                       target paths for each target domain to crawl
 * @return a list of absolute urls
 */
const extractLinksToFollow = async (
    page: Page,
    targetWebsites: Record<string, string[]>,
): Promise<string[]> => {
    // Find all the anchor tags and get the url from each
    const urls = await page.$$eval('a', (elements => elements.map(e => e.getAttribute('href'))));

    // Get the base url for any relative urls
    const currentPageUrlParts = (await page.evaluate(() => document.location.href)).split('/');
    const relativeUrlBase = currentPageUrlParts.slice(0, currentPageUrlParts.length).join('/');

    const filteredUrls = [];
    for (const url of urls) {
        if (url === null) {
            continue;
        }

        const u = new URL(url!, relativeUrlBase);

        // Filter to only urls within our target websites
        if (u.hostname in targetWebsites) {
            for (const targetPath of targetWebsites[u.hostname]) {
                if (u.pathname.startsWith(targetPath)) {
                    filteredUrls.push(u.href);
                    break;
                }
            }
        }
    }

    return Array.from(new Set(filteredUrls));
};

/**
 * Uses the given browser to load the given page, writes its content
 * to the destination, and returns any urls discovered from the page.
 * 
 * @param browser the puppeteer browser
 * @param input the page to visit
 * @param destination the location to write content to
 * @return a list of urls that were found on the page
 */
export const extractPageContentAndUrls = async (
    browser: Browser,
    input: CrawlPageInput,
    destination: CrawlDestination,
): Promise<string[]> => {
    const url = input.url;
    const pageID = getHashHex(url);

    try {
        // Visit the url and wait until network settles;
        // a reasonable indication that js libraries etc have all loaded and
        // client-side rendering or ajax calls have completed
        const page = await browser.newPage();
        await page.goto(url, {
            timeout: 10000,
            waitUntil: 'networkidle2',
        });

        // Extract the content from the page
        const content = await extractContent(page);

        // Write the content to the given destination
        await writePageContent(url, content, pageID, destination);

        // Extract fully qualified urls
        const discoveredUrls = await extractLinksToFollow(page, input.targetWebsites);

        return discoveredUrls;
    } catch (err) {
        console.warn('Crawl error:', url, err);
        return [];
    }
};
