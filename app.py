from flask import Flask, request, jsonify
from transformers import pipeline
import os

os.environ['TRANSFORMERS_CACHE'] = '/tmp/huggingface_cache'


app = Flask(__name__)

# Initialize the zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", revision="c626438")

@app.route('/test', methods=['GET'])

def hello():
    return "Hello World"

@app.route('/classify', methods=['POST'])
def classify():
    data = request.get_json()
    content = data.get('Content')
    labels = data.get('labels', [])

    if not content or not labels:
        return jsonify({'error': 'Invalid input'}), 400

    # Perform zero-shot classification
    result = classifier(content, labels)
    top_label = result['labels'][0]  # Get the most likely label

    # Include the original data in the response with the classification result
    response = {
        "ID": data.get("ID"),
        "URL": data.get("URL"),
        "Heading": data.get("Heading"),
        "Content": data.get("Content"),
        "Source": data.get("Source"),
        "Type": data.get("Type"),
        "Likes": data.get("Likes"),
        "Comments": data.get("Comments"),
        "Shares": data.get("Shares"),
        "Datetime": data.get("Datetime"),
        "Classified_Label": top_label
    }

    return jsonify(response), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)  # Run on a different port
