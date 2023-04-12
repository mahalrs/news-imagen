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

/**
 * Base configuration required for crawling
 */
export interface CrawlConfig {
    targetWebsites: Record<string, string[]>;
}

/**
 * Input required to start a web crawl
 */
export interface CrawlInput extends CrawlConfig {
    crawlName: string;
    crawlId: string;
    startUrls: string[];
}

/**
 * Input required to crawl an individual page
 */
export interface CrawlPageInput extends CrawlConfig {
    url: string;
}

/**
 * Destination for storing crawled content
 */
export interface CrawlDestination {
    dirPath: string;
}

/**
 * Represents the content of a web page
 */
export interface PageContent {
    title: string;
    htmlContent: string;
}
