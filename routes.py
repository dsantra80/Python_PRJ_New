from flask import Blueprint, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

main = Blueprint('main', __name__)

# Hugging Face API configuration
model_id = "gpt2"  # Replace this with your model_id once verified
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

if not huggingface_token:
    raise EnvironmentError("HUGGINGFACE_TOKEN environment variable is not set")

# Load model and tokenizer with authorization token
tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=huggingface_token)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    use_auth_token=huggingface_token
)

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    user_message = data['prompt']
    
    messages = [
        {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        {"role": "user", "content": user_message},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("")
    ]

    max_tokens = int(os.getenv("MAX_TOKENS", 256))
    temperature = float(os.getenv("TEMPERATURE", 0.6))
    
    outputs = model.generate(
        input_ids,
        max_new_tokens=max_tokens,
        eos_token_id=terminators,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
    )
    
    response = outputs[0][input_ids.shape[-1]:]
    generated_text = tokenizer.decode(response, skip_special_tokens=True)
    return jsonify({'generated_text': generated_text})
