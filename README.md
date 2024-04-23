CS429 - INFORMATION RETRIVAL PROJECT
GURUKIRAN SHIVASHANKAR
CWID : A20564280

# Abstract:

The Processor Search Engine project offers a comprehensive solution to the challenges of information retrieval in today's digital landscape. By seamlessly integrating web content crawling, indexing, and query processing functionalities, the project aims to revolutionize the way users access and retrieve online information. Using Python 3.10+ alongside Scrapy, Scikit-Learn, and Flask libraries, it represents a paradigm shift in information access and retrieval.

The project utilizes Scrapy for web crawling, systematically collecting HTML-formatted content with parameters such as seed URL/Domain, Max Pages, and Max Depth. The harvested data is then indexed using Scikit-Learn, constructing an inverted index based on TF-IDF score representation and cosine similarity computation.

The primary objective of the project is to develop a robust search engine capable of efficiently retrieving relevant information from the vast expanse of the web. This involves implementing advanced crawling, indexing, and query processing techniques to ensure accurate and timely results for user queries.

Moving forward, the project aims to enhance its capabilities with real-time indexing updates, query expansion, and personalized search. Efforts will focus on optimizing performance and scalability to meet evolving user demands and web content structures.

# Overview:

**Solution Outline:**

- The Processor Search Engine aims to develop a comprehensive search solution integrating web content crawling, indexing, and query processing functionalities.
- Leveraging Python 3.10+ alongside Scrapy, Scikit-Learn, and Flask libraries, the system represents a significant advancement in information retrieval technology.
- By systematically collecting HTML-formatted content from web domains and constructing an inverted index based on TF-IDF score representation and cosine similarity computation, the Processor Search Engine streamlines information access and retrieval.

**Relevant Literature:**

- The project builds upon existing literature in the fields of information retrieval, web crawling, and search engine technologies.
- Key concepts include TF-IDF representation, cosine similarity scoring, web crawling techniques, and indexing algorithms.
- Notable works by Salton and Buckley ("Term-weighting Approaches in Automatic Text Retrieval") and Manning, Raghavan, and Schütze ("Introduction to Information Retrieval") provide foundational knowledge.

**Proposed System:**

- The proposed system leverages modern web technologies and libraries such as Scrapy, Scikit-Learn, and Flask to create a robust and efficient search engine.
- By integrating advanced techniques like TF-IDF representation and Cosine similarity scoring, the system aims to deliver accurate and relevant search results to users, thereby improving user satisfaction and engagement.
- The Processor Search Engine offers a streamlined approach to information access, allowing users to retrieve pertinent information from a vast collection of web content with ease.

# DESIGN :

**System Capabilities**:

Crawler: The system utilizes Scrapy, a powerful web crawling framework, to systematically traverse web domains and collect HTML-formatted content. With parameters such as seed URL/domain, maximum pages, and maximum depth, the crawler ensures targeted exploration of the web, enabling efficient data acquisition.

Indexer: Leveraging Scikit-Learn, the system constructs an inverted index based on TF-IDF score representation and cosine similarity computation. This enables rapid and accurate retrieval of relevant information, providing users with a seamless and intuitive search experience.

Processor: Developed using Flask, the processor handles HTTP requests for query processing. It validates and checks errors in free text queries in JSON format and returns top-K ranked results based on relevance to the query.

**Interactions**:

- The Crawler interacts with web pages to extract relevant content and metadata, including URLs and titles.
- The Indexer processes the crawled HTML documents to build an inverted index, enabling efficient search indexing based on TF-IDF representation and cosine similarity.
- The Processor handles user queries, validating and error-checking them before returning top-K ranked results, providing an interactive search experience.

**Integration:**

- The Crawler, Indexer, and Processor components seamlessly integrate to form the Processor Search Engine, enabling end-to-end web content crawling, indexing, and query processing functionalities.
- Utilizing Python 3.10+ alongside Scrapy, Scikit-Learn, and Flask libraries, the system ensures compatibility and efficiency in its design and implementation.

# Architecture:

**Software Components:**

- Crawler: Implemented using the Scrapy framework.
- Indexer: Utilizes the Scikit-Learn library.
- Processor: Developed using the Flask framework.

**Interfaces:**

Crawler Interface: Accepts parameters such as seed URL/domain, maximum pages, and maximum depth for crawling initiation.

Indexer Interface: Interfaces with the output of the crawler component to process crawled HTML documents and build the inverted index.

Processor Interface: Receives user queries in JSON format via HTTP requests, performs query validation and processing, and returns search results.

**Implementation:**

Crawler Implementation: Utilizes the Scrapy framework to define spiders and pipelines for crawling web content. Implements logic for managing crawling parameters and extracting content and metadata from crawled pages.

Indexer Implementation: Utilizes the Scikit-Learn library to vectorize text data and compute TF-IDF scores. Implements logic for building the inverted index structure based on computed scores.

Processor Implementation: Implements HTTP endpoints using the Flask framework to handle user queries. Implements logic for query validation, execution, and result retrieval.

# Operation:

**Software Commands:**

Crawler: Run the crawler using the command “**scrapy crawl posts”** from the command line.

Indexer: Execute the main() function in the Indexer script to initialize indexing and search functionalities. Use command **“python Indexer.py”**

Processor: Run the Flask application using the command “**python app.py”** from the command line to start the processor.

**Inputs:**

Crawler: Parameters such as seed URL/domain, maximum pages, and maximum depth are inputted to initialize crawling.

Indexer: Requires the HTML file generated by the crawler as input for indexing.

Processor: Accepts free text queries in JSON format via HTTP requests.

**Installation:**

- Install Python 3.10+.
- Install required libraries using pip install scrapy scikit-learn flask.
- Ensure the Scrapy, Scikit-Learn, and Flask libraries are installed correctly.
- Run the provided scripts according to the specified commands.

# Conclusion:

The Processor Search Engine project concludes with insights into its success/failure results, outputs, and potential caveats/cautions.

**Success/Failure Results**:

Success: The project successfully implements web content crawling, indexing, and query processing functionalities, providing users with an efficient search experience.

Failure: Potential failures may arise due to issues such as network connectivity problems, malformed HTML content.

**Outputs:**

Crawler Output: HTML files containing crawled web content.

Indexer Output: Inverted index structure stored in pickle format.

Processor Output: Top-K ranked search results returned as JSON objects.

**Caveats/Cautions:**

Data Quality: The effectiveness of the search engine depends on the quality and relevance of the crawled data.

Resource Consumption: Crawling large volumes of data may consume significant system resources and bandwidth.

Security Considerations: Ensure proper validation and sanitization of user input to prevent security vulnerabilities such as injection attacks.

# Output Screenshots :

**Output 1 :  A scrapy crawler for downloading web document in HTML format -content**

Requirement : Initialize using seed URL/Domain, Max Pages, Max Depth.

![Screenshot 2024-04-22 203858](Aspose.Words.740d99e5-3d50-4df4-8e15-099f3fa04e4b.001.png)

**Output 2 : A Scikit-Learn based Indexer for contructing an inverted index in pickle format - search indexing**

Requirement : TF-IDF score/weight representation, Cosine similarity

![Screenshot 2024-04-22 204103](Aspose.Words.740d99e5-3d50-4df4-8e15-099f3fa04e4b.002.png)

![Screenshot 2024-04-22 204114](Aspose.Words.740d99e5-3d50-4df4-8e15-099f3fa04e4b.003.png)

![Screenshot 2024-04-22 204138](Aspose.Words.740d99e5-3d50-4df4-8e15-099f3fa04e4b.004.png)

**Output 3 : A Flask based Processor for handling free text queries in json format - query processing**

Required: Query validation/error-checking, Top-K ranked results

![Screenshot 2024-04-22 204224](Aspose.Words.740d99e5-3d50-4df4-8e15-099f3fa04e4b.005.png)

![Screenshot 2024-04-22 204318](Aspose.Words.740d99e5-3d50-4df4-8e15-099f3fa04e4b.006.png)

![Screenshot 2024-04-22 204328](Aspose.Words.740d99e5-3d50-4df4-8e15-099f3fa04e4b.007.png)



# Data Sources:

**Seed URL/Domain**: In the provided code, the seed URL used is **'https://www.gutenberg.org/'**. This URL initiates the crawling process, and the crawler recursively traverses the web domain to collect HTML-formatted content for indexing and processing.

**Flask Server**: Users can interact with the Processor component by sending HTTP requests to the specified endpoints, typically via the server's IP address and port number (e.g., <http://localhost:5000>).

# Test Cases

**Inverted Index Construction Test:**

Ensure the indexer constructs an inverted index with the expected TF-IDF score representation.

Framework: Manual testing.

Harness: None required.

Coverage: Inverted index construction, TF-IDF score representation.

Command: **python Indexer.py**

**Cosine Similarity Test:**

Test the functionality of cosine similarity computation for search indexing.

Framework: Manual testing.

Harness: None required.

Coverage: Cosine similarity computation.

**Search Result Validation Test:**

Validate the accuracy of search results retrieved from the index.

Framework: Manual testing.

Harness: None required.

Coverage: Search result accuracy.

**Processor:**

HTTP Endpoint Testing:

Test HTTP endpoints to ensure that users can interact with the Processor component.

Framework: Manual testing.

Harness: None required.

Coverage: HTTP endpoint functionality.

Command: python app.py

**Query Processing Test:**

Validate that the Processor handles query processing accurately and returns top-K ranked results.

Framework: Manual testing.

Harness: None required.

Coverage: Query processing accuracy, top-K ranked results.

**Integration Testing:**

Conduct integration testing to ensure seamless communication between the Processor and other components.

Framework: Manual testing.

Harness: None required.

Coverage: Integration with other components.

**Commands:**

scrapy crawl posts (Crawler)

python Indexer.py (Indexer)

python app.py (Processor)


# Source Code :

**Part 1 :**

**HTML ans Json Crawling Program :** 

import scrapy

import json

class PostaSpider(scrapy.Spider):

`    `name = "posts"

`    `max\_pages\_to\_store = 15  # Maximum number of pages to store in the output file

`    `max\_depth = 10  # Maximum depth of crawling

`    `start\_urls = [

`        `"https://www.gutenberg.org/"

`    `]

`    `def \_\_init\_\_(self, \*args, \*\*kwargs):

`        `super(PostaSpider, self).\_\_init\_\_(\*args, \*\*kwargs)

`        `self.visited\_urls = set()

`        `self.output\_html\_file = open('output.html', 'w', encoding='utf-8')  # Open file in text mode with UTF-8 encoding

`        `self.output\_json\_file = open('output.json', 'w')

`        `self.pages\_crawled = 0

`        `self.data = []  # List to store crawled data

`    `def parse(self, response):

`        `if self.pages\_crawled >= self.max\_pages\_to\_store:

`            `self.logger.info('Maximum pages to store limit reached.')

`            `self.close\_files()  # Close the output files

`            `return

`        `if response.url in self.visited\_urls:

`            `return

`        `self.visited\_urls.add(response.url)

`        `# Append response body to the output HTML file

`        `if self.pages\_crawled < self.max\_pages\_to\_store:

`            `self.output\_html\_file.write(response.text)  # Write HTML content as text

`            `self.pages\_crawled += 1

`            `# Extract data and append to self.data

`            `data = {

`                `'url': response.url,

`                `'title': response.css('title::text').get(),

`                `# Add more data fields as needed

`            `}

`            `self.data.append(data)

`        `current\_depth = response.meta.get('depth', 1)

`        `if current\_depth >= self.max\_depth:

`            `self.logger.info('Maximum depth limit reached at depth %d.' % current\_depth)

`            `return

`        `for next\_page in response.css('a::attr(href)').extract():

`            `if self.pages\_crawled >= self.max\_pages\_to\_store:

`                `self.logger.info('Maximum pages to store limit reached.')

`                `self.close\_files()  # Close the output files

`                `return

`            `yield response.follow(next\_page, callback=self.parse, meta={'depth': current\_depth + 1})

`    `def closed(self, reason):

`        `self.close\_files()

`    `def close\_files(self):

`        `self.output\_html\_file.close()

`        `# Convert crawled data to JSON and write to file

`        `try:

`            `with open('output.json', 'w') as json\_file:

`                `json.dump(self.data, json\_file, indent=4)

`        `except Exception as e:

`            `self.logger.error(f"Failed to write JSON data to file: {str(e)}")

`        `else:

`            `self.logger.info("JSON data successfully written to file.")

\# Run spider from command line

\# code to run : scrapy crawl posts

**Part - 2**

**INDEXER :** 

import os

import pickle

import nltk

import json

from sklearn.feature\_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine\_similarity

from nltk.tokenize import word\_tokenize

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer, WordNetLemmatizer

import warnings

warnings.filterwarnings("ignore")

nltk.download('wordnet')

nltk.download('omw-1.4')

class Indexer:

`    `def \_\_init\_\_(self, html\_file\_path):

`        `self.html\_file\_path = html\_file\_path

`        `self.vectorizer = TfidfVectorizer(stop\_words=None, tokenizer=self.custom\_tokenizer, token\_pattern=None)

`        `self.documents, self.filenames = self.retrieve\_documents()

`        `self.inverted\_index = {}  

`        `if self.documents:

`            `self.tfidf\_matrix = self.vectorizer.fit\_transform(self.documents)  # Fit the vectorizer here

`            `self.inverted\_index = self.build\_inverted\_index()

`        `else:

`            `self.tfidf\_matrix = None

`            `print("No documents found.")

`        `self.save\_inverted\_index\_to\_pickle()  # Print inverted index when initializing            

`    `def retrieve\_documents(self):

`        `documents = []

`        `filenames = []

`        `with open(self.html\_file\_path, 'r', encoding="utf-8", errors="ignore") as html\_file:

`            `text = html\_file.read()  # Read the entire content of the HTML file

`            `documents.append(text)

`            `filenames.append(self.html\_file\_path)  # Using the HTML file path as the filename

`        `return documents, filenames    

`    `def build\_inverted\_index(self):

`        `inverted\_index = {}

`        `terms = self.vectorizer.get\_feature\_names\_out()

`        `term\_indices = {term: i for i, term in enumerate(terms)}

`        `for i, doc in enumerate(self.documents):

`            `for term in terms:

`                `if term in doc:

`                    `if term not in inverted\_index:

`                        `inverted\_index[term] = []

`                    `inverted\_index[term].append(i)

`        `return inverted\_index    

`    `def custom\_tokenizer(self, text):

`        `stop\_words = set(stopwords.words('english'))

`        `stemmer = PorterStemmer()

`        `lemmatizer = WordNetLemmatizer()

`        `tokens = word\_tokenize(text)

`        `tokens = [stemmer.stem(token) for token in tokens if token not in stop\_words]

`        `tokens = [lemmatizer.lemmatize(token) for token in tokens]

`        `return tokens

`    `def search(self, query, top\_k=5):

`        `if self.tfidf\_matrix is None or self.tfidf\_matrix.shape[0] == 0:

`            `return []

`        `query\_vector = self.vectorizer.transform([query])

`        `query\_terms = query.split()

`        `query\_terms\_set = set(query\_terms)

`        `cosine\_similarities = cosine\_similarity(query\_vector, self.tfidf\_matrix)

`        `scores = cosine\_similarities.flatten()

`        `doc\_indices = scores.argsort()[::-1]

`        `result = []

`        `for doc\_index in doc\_indices[:top\_k]:  # Select top-K results

`            `filename = self.filenames[doc\_index]

`            `score = scores[doc\_index]

`            `if score > 0:

`                `tfidf\_scores = {term: self.tfidf\_matrix[doc\_index, self.vectorizer.vocabulary\_.get(term, -1)] for term in query\_terms\_set}

`                `result.append((filename, score, tfidf\_scores))

`        `return result

`    `def save\_search\_results\_to\_pickle(self, results, filename):

`        `with open(filename, 'wb') as f:

`            `pickle.dump(results, f, protocol=pickle.HIGHEST\_PROTOCOL)

`        `print("Search results:")

`        `for filename, similarity\_score, tfidf\_scores in results:

`            `print(f" Cosine Similarity Score: {similarity\_score}, TF-IDF Scores: {tfidf\_scores}")

`    `def save\_inverted\_index(self):

`        `print("Inverted index:")

`        `print("{")

`        `for term, indices in self.inverted\_index.items():

`            `print(f"'{term}': {indices}, ", end="")

`        `print("\n}")

`    `def save\_inverted\_index\_to\_pickle(self):

`        `print("Inverted index content (Pickle format):")

`        `print(pickle.dumps(self.inverted\_index))  # Print content in pickle format directly to terminal

`        `self.save\_inverted\_index()

def main():

`    `html\_file\_path='E:\\Masters\\Information retrival\\project\\IRR (2)\\IRR\\IRR\\project\\posts\\output.html'    

`    `indexer = Indexer(html\_file\_path)

`    `query = input("Enter your query: ")

`    `results = indexer.search(query)

`    `indexer.save\_search\_results\_to\_pickle(results, "search\_results.pickle")

if \_\_name\_\_ == "\_\_main\_\_":

main()

**Part - 3 :**

**FLASK :**

from flask import Flask, request, jsonify, render\_template

import json  # Import the json module

from sklearn.feature\_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine\_similarity

app = Flask(\_\_name\_\_)

\# Initialize global variables

documents = []

urls = []

titles = []

tfidf\_matrix = None

vectorizer = None

\# Function to retrieve documents from the JSON file

def retrieve\_documents(json\_file\_path):

`    `try:

`        `with open(json\_file\_path, 'r') as json\_file:

`            `data = json.load(json\_file)

`            `for entry in data:

`                `title = entry.get('title', '')  # Fetch title from JSON

`                `url = entry.get('url', '')  # Fetch URL from JSON

`                `if title.strip() and url.strip():

`                    `documents.append(title)  # Append title to documents list

`                    `titles.append(title)  # Append title to titles list

`                    `urls.append(url)  # Append URL to urls list

`            `if not documents:

`                `raise ValueError("No valid documents found in the JSON file.")

`    `except FileNotFoundError:

`        `raise FileNotFoundError(f"JSON file not found at path: {json\_file\_path}")

`    `except json.JSONDecodeError:

`        `raise ValueError(f"Invalid JSON format in file: {json\_file\_path}")

\# Function to initialize TF-IDF vectorizer and matrix

def initialize\_vectorizer():

`    `global vectorizer, tfidf\_matrix

`    `vectorizer = TfidfVectorizer(stop\_words='english')

`    `tfidf\_matrix = vectorizer.fit\_transform(documents)

\# Function to perform search

def search(query, top\_k=5):

`    `global tfidf\_matrix

`    `if tfidf\_matrix is None:

`        `raise ValueError("Vectorizer is not initialized. Call initialize\_vectorizer() first.")    

`    `query\_vector = vectorizer.transform([query])

`    `cosine\_similarities = cosine\_similarity(query\_vector, tfidf\_matrix)

`    `scores = cosine\_similarities.flatten()

`    `doc\_indices = scores.argsort()[::-1]

`    `result = []

`    `for doc\_index in doc\_indices[:top\_k]:  # Select top-K results

`        `url = urls[doc\_index]

`        `title = titles[doc\_index]

`        `score = scores[doc\_index]

`        `if score > 0:

`            `result.append((url, title, score))

`    `return result

\# Path to the output.json file generated by the crawling process

json\_file\_path= "E:\\Masters\\Information retrival\\project\\IRR (2)\\IRR\\IRR\\project\\posts\\output.json"

try:

`    `# Retrieve documents from the JSON file

`    `retrieve\_documents(json\_file\_path)

`    `# Initialize TF-IDF vectorizer and matrix

`    `initialize\_vectorizer()

except Exception as e:

`    `print(f"An error occurred: {e}")

@app.route('/', methods=['GET'])

def index():

`    `# Render the index.html template

`    `return render\_template('index.html')

@app.route('/search', methods=['GET'])

def search\_endpoint():

`    `# Get the query parameter from the request

`    `query = request.args.get('query')



`    `# If no query is provided, return an error response

`    `if not query:

`        `return jsonify({"error": "No query provided"}), 400  

`  `try:

`        `# Perform search

`        `results = search(query)        

`        `# If results are found, render the results template with the results

`        `if results:

`            `return render\_template('results.html', results=results, query=query), 200

`        `else:

`            `# If no results are found, return a message indicating so

`            `return jsonify({"message": "No results found"}), 404

`    `except Exception as e:

`        `return jsonify({"error": str(e)}), 500

if \_\_name\_\_ == '\_\_main\_\_':

`    `app.run(debug=True)



# Documentation:

**Part 1: Crawling**

Description : This part of the code defines a web crawler using Scrapy to extract data from web pages.

Dependencies:

Scrapy: A Python framework for web crawling and scraping.

json: A built-in Python library for JSON manipulation.

Usage: Run the spider using the command “scrapy crawl posts”.

Configuration:

max\_pages\_to\_store: Maximum number of pages to store in the output file.

max\_depth: Maximum depth of crawling.

Usage Example: An example usage is provided at the end of the source code.

**Part 2: Indexer**

Description: This part of the code defines an indexer to build an inverted index from HTML documents.

Dependencies:

os: A built-in Python library for operating system-related functions.

pickle: A built-in Python library for serializing and deserializing Python objects.

nltk: Natural Language Toolkit for natural language processing tasks.

sklearn: Scikit-learn library for machine learning algorithms.

Usage: Initialize the indexer with the HTML file path and perform searches using the search() method. 

\- Run the code using the command “python Indexer.py”

Configuration:

html\_file\_path: Path to the HTML file generated by the crawling process.

**Part 3: Flask**

Description: This part of the code defines a Flask web application for searching through indexed 

document.

Dependencies:

Flask: A Python web framework.

sklearn: Scikit-learn library for machine learning algorithms.

Usage: Start the Flask application and access the search interface through a web browser.

Configuration:

json\_file\_path: Path to the JSON file containing crawled data.

# Open-Source Dependencies:

Scrapy: A powerful web crawling and web scraping framework written in Python.

Website: https://scrapy.org/

GitHub Repository: https://github.com/scrapy/scrapy

NLTK: Natural Language Toolkit is a leading platform for building Python programs to work with human language data. 

Website: https://www.nltk.org/

GitHub Repository: https://github.com/nltk/nltk

Scikit-learn: A simple and efficient tools for predictive data analysis, built on NumPy, SciPy, and matplotlib.

Website: https://scikit-learn.org/

GitHub Repository: <https://github.com/scikit-learn/scikit-learn>

# Bibliography : 

- ChatGPT. "ChatGPT." OpenAI. Accessed April 15, 2024. https://openai.com/chatgpt.
- Schafer, Corey. "Scrapy for Beginners - A Complete How To Example Web Scraping Project." Uploaded by Corey Schafer. September 10, 2018. YouTube video, 22:44. Accessed April 15, 2024.[Online] https://youtu.be/s4jtkzHhLzY.
- Corey Schafer, "JSON with Flask - Python on the web - Learning Flask Series Pt. 9," Corey Schafer, Mar. 10, 2017. [Online]. Available: https://youtu.be/VzBtoA_8qm4.
