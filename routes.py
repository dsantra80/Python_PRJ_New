from flask import Blueprint, render_template, request, jsonify
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import os

main = Blueprint('main', __name__)

# Load the text generation pipeline
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
pipe = pipeline("text-generation", model=model_id, tokenizer=model_id)

# Load model and tokenizer directly
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

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
