from flask import Flask, request, jsonify, render_template
import json  # Import the json module
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Initialize global variables
documents = []
urls = []
titles = []
tfidf_matrix = None
vectorizer = None

# Function to retrieve documents from the JSON file
def retrieve_documents(json_file_path):
    try:
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)
            for entry in data:
                title = entry.get('title', '')  # Fetch title from JSON
                url = entry.get('url', '')  # Fetch URL from JSON
                if title.strip() and url.strip():
                    documents.append(title)  # Append title to documents list
                    titles.append(title)  # Append title to titles list
                    urls.append(url)  # Append URL to urls list
            if not documents:
                raise ValueError("No valid documents found in the JSON file.")
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file not found at path: {json_file_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in file: {json_file_path}")

# Function to initialize TF-IDF vectorizer and matrix
def initialize_vectorizer():
    global vectorizer, tfidf_matrix
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)

# Function to perform search
def search(query, top_k=5):
    global tfidf_matrix
    if tfidf_matrix is None:
        raise ValueError("Vectorizer is not initialized. Call initialize_vectorizer() first.")
    
    query_vector = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix)
    scores = cosine_similarities.flatten()
    doc_indices = scores.argsort()[::-1]
    result = []
    for doc_index in doc_indices[:top_k]:  # Select top-K results
        url = urls[doc_index]
        title = titles[doc_index]
        score = scores[doc_index]
        if score > 0:
            result.append((url, title, score))
    return result

# Path to the output.json file generated by the crawling process
json_file_path = "C:\\Users\\Gurukiran S\\OneDrive\\Desktop\\ir\\project\\Scrapy\\posts\\output.json"

try:
    # Retrieve documents from the JSON file
    retrieve_documents(json_file_path)

    # Initialize TF-IDF vectorizer and matrix
    initialize_vectorizer()
except Exception as e:
    print(f"An error occurred: {e}")

@app.route('/', methods=['GET'])
def index():
    # Render the index.html template
    return render_template('index.html')

@app.route('/search', methods=['GET'])
def search_endpoint():
    # Get the query parameter from the request
    query = request.args.get('query')
    
    # If no query is provided, return an error response
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    try:
        # Perform search
        results = search(query)
        
        # If results are found, render the results template with the results
        if results:
            return render_template('results.html', results=results, query=query), 200
        else:
            # If no results are found, return a message indicating so
            return jsonify({"message": "No results found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
