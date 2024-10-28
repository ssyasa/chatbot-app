import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import time

# Load environment variables from the .env file
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Get the Hugging Face API key from environment variables
api_key = os.getenv("HUGGINGFACE_API_KEY")
if not api_key:
    raise ValueError("API key for Hugging Face not found in .env file")

# Initialize the InferenceClient with the API key
client = InferenceClient(api_key=api_key)

# Rate limiting variables
last_request_time = time.time()
request_interval = 2.0  # Set interval to 2 seconds or more to prevent spamming

def format_response(response):
    """Format the response into structured HTML."""
    paragraphs = response.strip().split('\n')
    structured_text = ""

    for paragraph in paragraphs:
        if paragraph.startswith('-'):
            structured_text += f"<li>{paragraph[1:].strip()}</li>"  # Add list item
        else:
            structured_text += f"<p>{paragraph.strip()}</p>"

    if "<li>" in structured_text:
        structured_text = "<ul>" + structured_text + "</ul>"
    
    return structured_text

def get_llama_response(message):
    global last_request_time

    current_time = time.time()
    if current_time - last_request_time < request_interval:
        return "Error: Too many requests. Please wait a moment."

    messages = [
        {"role": "user", "content": message}
    ]

    try:
        stream = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            messages=messages,
            max_tokens=500,
            stream=True
        )

        response_content = ""
        for chunk in stream:
            response_content += chunk.choices[0].delta.content

        last_request_time = current_time
        return format_response(response_content)
        
    except Exception as e:
        return f"An error occurred: {str(e)}"  # Make sure this line is properly indented

@app.route('/')
def index():
    return render_template('index.html')  # Render the main HTML page

@app.route('/llama', methods=['POST'])
def llama_chat():
    user_input = request.json.get('user_input')  # Fetch user input from JSON body
    if not user_input:
        return jsonify(content="Error: No input provided."), 400

    content = get_llama_response(user_input)
    return jsonify(content=content)

if __name__ == '__main__':
    app.run(debug=True, port=8080, host='0.0.0.0')
