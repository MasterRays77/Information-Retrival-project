import scrapy
import json

class PostaSpider(scrapy.Spider):
    name = "posts"
    max_pages_to_store = 15  # Maximum number of pages to store in the output file
    max_depth = 10  # Maximum depth of crawling

    start_urls = [
        "https://www.gutenberg.org/"
    ]

    def __init__(self, *args, **kwargs):
        super(PostaSpider, self).__init__(*args, **kwargs)
        self.visited_urls = set()
        self.output_html_file = open('output.html', 'w', encoding='utf-8')  # Open file in text mode with UTF-8 encoding
        self.output_json_file = open('output.json', 'w')
        self.pages_crawled = 0
        self.data = []  # List to store crawled data

    def parse(self, response):
        if self.pages_crawled >= self.max_pages_to_store:
            self.logger.info('Maximum pages to store limit reached.')
            self.close_files()  # Close the output files
            return

        if response.url in self.visited_urls:
            return

        self.visited_urls.add(response.url)

        # Append response body to the output HTML file
        if self.pages_crawled < self.max_pages_to_store:
            self.output_html_file.write(response.text)  # Write HTML content as text
            self.pages_crawled += 1

            # Extract data and append to self.data
            data = {
                'url': response.url,
                'title': response.css('title::text').get(),
                # Add more data fields as needed
            }
            self.data.append(data)

        current_depth = response.meta.get('depth', 1)
        if current_depth >= self.max_depth:
            self.logger.info('Maximum depth limit reached at depth %d.' % current_depth)
            return

        for next_page in response.css('a::attr(href)').extract():
            if self.pages_crawled >= self.max_pages_to_store:
                self.logger.info('Maximum pages to store limit reached.')
                self.close_files()  # Close the output files
                return
            yield response.follow(next_page, callback=self.parse, meta={'depth': current_depth + 1})

    def closed(self, reason):
        self.close_files()

    def close_files(self):
        self.output_html_file.close()

        # Convert crawled data to JSON and write to file
        try:
            with open('output.json', 'w') as json_file:
                json.dump(self.data, json_file, indent=4)
        except Exception as e:
            self.logger.error(f"Failed to write JSON data to file: {str(e)}")
        else:
            self.logger.info("JSON data successfully written to file.")

# Run spider from command line
# scrapy crawl posts
