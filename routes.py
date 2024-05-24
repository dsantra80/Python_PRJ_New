from flask import Blueprint, render_template, request, jsonify
import transformers
import torch
import os

main = Blueprint('main', __name__)

# Hugging Face API configuration
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

if not huggingface_token:
    raise EnvironmentError("HUGGINGFACE_TOKEN environment variable is not set")

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    use_auth_token=huggingface_token,
    device_map="auto",
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

    prompt = pipeline.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("")
    ]

    max_tokens = int(os.getenv("MAX_TOKENS", 256))
    temperature = float(os.getenv("TEMPERATURE", 0.6))
    
    outputs = pipeline(
        prompt,
        max_new_tokens=max_tokens,
        eos_token_id=terminators,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
    )

    generated_text = outputs[0]["generated_text"][len(prompt):]
    return jsonify({'generated_text': generated_text})
