from flask import Blueprint, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
import os

main = Blueprint('main', __name__)

# Hugging Face API configuration
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

if not huggingface_token:
    raise EnvironmentError("HUGGINGFACE_TOKEN environment variable is not set")

# Load model and tokenizer directly with authorization token
tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=huggingface_token)
model = AutoModelForCausalLM.from_pretrained(model_id, use_auth_token=huggingface_token)

# Create text generation pipeline
pipe = TextGenerationPipeline(model=model, tokenizer=tokenizer)

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data['prompt']
    max_tokens = int(os.getenv("MAX_TOKENS", 100))
    temperature = float(os.getenv("TEMPERATURE", 1.0))
    
    # Generate text using the pipeline
    output = pipe(prompt, max_length=max_tokens, temperature=temperature)
    generated_text = output[0]['generated_text']
    return jsonify({'generated_text': generated_text})
