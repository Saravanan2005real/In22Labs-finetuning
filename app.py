import os
import sys
import json
import string
from datasets import Dataset
import torch
import argparse
from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

app = Flask(__name__)

# Parse command line arguments
parser = argparse.ArgumentParser(description="Legal Chatbot Server")
parser.add_argument("--mock", action="store_true", help="Run with mock LLM responses for quick UI testing")
parser.add_argument("--port", type=int, default=5000, help="Server port number")
args, unknown = parser.parse_known_args()

# Check environment variable for mock mode as well
if os.environ.get("MOCK_MODE", "").lower() in ("true", "1", "yes"):
    args.mock = True

# Global variables for model, tokenizer, and cached sections
model = None
tokenizer = None
device = "cpu"
SECTIONS_CACHE = None

def load_sections():
    global SECTIONS_CACHE
    if SECTIONS_CACHE is not None:
        return SECTIONS_CACHE
    sections_path = "sections.json"
    if os.path.exists(sections_path):
        try:
            with open(sections_path, "r", encoding="utf-8") as f:
                SECTIONS_CACHE = json.load(f)
                print(f"Successfully loaded {len(SECTIONS_CACHE)} sections in-memory.")
                return SECTIONS_CACHE
        except Exception as e:
            print(f"Error reading sections.json: {e}")
    else:
        print("WARNING: sections.json not found. Run extraction script first.")
    SECTIONS_CACHE = []
    return SECTIONS_CACHE

# Load model and tokenizer
if args.mock:
    print("\n--- RUNNING IN MOCK MODE ---")
    print("Model loading skipped. Chat responses will be mocked.")
else:
    print("\nLoading fine-tuned Llama model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Path to saved model
    adapter_path = "lora_model"
    base_model_name = "./llama_base_model"
    
    try:
        if os.path.exists(adapter_path) and os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
            print(f"Loading tokenizer from '{adapter_path}'...")
            tokenizer = AutoTokenizer.from_pretrained(adapter_path)
            
            if device == "cuda":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
                print("Loading base model in 4-bit on GPU...")
                model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    quantization_config=bnb_config,
                    device_map="auto"
                )
            else:
                print("Loading base model on CPU...")
                model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.float32,
                    device_map="cpu"
                )
            print("Model loaded successfully!")
        else:
            print(f"WARNING: '{adapter_path}' not found or invalid. Loading base model '{base_model_name}' without fine-tuning...")
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            
            if device == "cuda":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
                print("Loading base model in 4-bit on GPU...")
                model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    quantization_config=bnb_config,
                    device_map="auto"
                )
            else:
                print("Loading base model on CPU...")
                model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.float32,
                    device_map="cpu"
                )
            print("Base model loaded successfully!")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        print("Falling back to MOCK mode automatically.")
        args.mock = True

def tokenize(text):
    """
    Simple tokenizer for search matching. Preserves numeric tokens of any length.
    """
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return [w for w in text.split() if len(w) > 2 or w.isdigit()]

def search_legal_acts(query, limit=3):
    """
    Search legal acts using in-memory local sections database.
    """
    import re
    sections = load_sections()
    if not sections:
        return []
        
    query_tokens = tokenize(query)
    if not query_tokens:
        return []
        
    scored_sections = []
    for sec in sections:
        score = 0
        act_name = sec.get('act_name', '').lower()
        section_title = sec.get('section_title', '').lower()
        section_num = sec.get('section_num', '').lower()
        content = sec.get('content', '').lower()
        
        for token in query_tokens:
            pattern = r'\b' + re.escape(token) + r'\b'
            
            if token == section_num:
                score += 100
                
            # Act name matching (exact word vs substring)
            if re.search(pattern, act_name):
                score += 150
            else:
                if token in act_name:
                    score += 5
                    
            # Section title matching (exact word vs substring)
            if re.search(pattern, section_title):
                score += 50
            else:
                if token in section_title:
                    score += 10
                    
            # Content matching (exact word vs substring)
            if re.search(pattern, content):
                score += 5 * len(re.findall(pattern, content))
            else:
                if token in content:
                    score += 1
            
        if score > 0:
            scored_sections.append((score, sec))
            
    # Sort by score descending
    scored_sections.sort(key=lambda x: x[0], reverse=True)
    return [sec for score, sec in scored_sections[:limit]]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/status")
def status():
    """
    Returns the current status of Local DB and LLM Engine.
    """
    sections = load_sections()
    db_online = len(sections) > 0
    return jsonify({
        "database": db_online,
        "document_count": len(sections),
        "model": args.mock or (model is not None),
        "device": "MOCK" if args.mock else device
    })

@app.route("/api/acts")
def get_acts():
    """
    Returns unique acts from local dataset and their document counts.
    """
    sections = load_sections()
    if not sections:
        # Fallback to hardcoded list if empty for visual demo
        acts = [
            {"name": "Indian Partnership Act 1932", "count": 74},
            {"name": "Powers Of Attorney Act 1882", "count": 6},
            {"name": "Indian Christian Marriage Act 1872", "count": 88}
        ]
        return jsonify(acts)
        
    act_counts = {}
    for sec in sections:
        act_name = sec.get('act_name')
        if act_name:
            act_counts[act_name] = act_counts.get(act_name, 0) + 1
            
    acts = [{"name": name, "count": count} for name, count in act_counts.items()]
    acts.sort(key=lambda x: x['name'])
    return jsonify(acts)

@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Chat endpoint.
    Retrieves context from local database, builds prompt, and runs LLM inference.
    """
    import re
    data = request.json or {}
    query = data.get("query", "").strip()
    
    if not query:
        return jsonify({"response": "Please specify a query.", "context": []})
    
    # 1. Search relevant documents in local in-memory dataset
    context_docs = search_legal_acts(query)
    
    # Check if the query is a document-level summary/explanation query
    is_summary_query = any(keyword in query.lower() for keyword in ["summarize", "explain", "what is", "about", "overview"])
    has_specific_section = any(keyword in query.lower() for keyword in ["section ", "sec "])
    
    # If we found matches and it is a document-level summary query, fetch all sections of the matched act
    if context_docs and is_summary_query and not has_specific_section:
        top_act = context_docs[0]['act_name']
        all_sections = load_sections()
        act_sections = [s for s in all_sections if s['act_name'] == top_act]
        try:
            act_sections.sort(key=lambda s: int(s['section_num']) if s['section_num'].isdigit() else 999)
        except Exception:
            pass
        context_docs = act_sections
    
    # 2. Formulate RAG response
    if args.mock:
        # Clean, direct mock response generation with validation
        if context_docs:
            doc = context_docs[0]
            query_numbers = re.findall(r'\b\d+\b', query)
            is_valid = True
            if query_numbers:
                doc_title_text = f"{doc.get('act_name', '')} {doc.get('section_num', '')} {doc.get('section_title', '')}".lower()
                for num in query_numbers:
                    if len(num) <= 4: # G.O. numbers or section numbers are short
                        if num not in doc_title_text:
                            is_valid = False
                            break
            
            if is_valid:
                if is_summary_query and not has_specific_section and len(context_docs) > 1:
                    bullets = []
                    for s in context_docs:
                        bullets.append(f"- {s['content'].strip()}")
                    bullets_str = "\n".join(bullets)
                    response = (
                        f"G.O. (Ms.) No. {doc['act_name']} concerns the subject of this order. "
                        f"Here is a summary of the key provisions and decisions taken:\n\n{bullets_str}"
                    )
                else:
                    response = f"According to Section {doc['section_num']} ('{doc['section_title']}') of the {doc['act_name']}:\n\n{doc['content']}"
            else:
                response = "The provided context does not contain the answer for this question."
        else:
            response = "The provided context does not contain the answer for this question."
        return jsonify({"response": response, "context": context_docs})

    # 3. LLM Inference
    try:
        if context_docs:
            # Build structured context string
            context_str = ""
            for i, doc in enumerate(context_docs):
                context_str += f"Source #{i+1}:\nAct: {doc['act_name']}\nSection: {doc['section_num']} - {doc['section_title']}\nText: {doc['content']}\n\n"
            
            prompt = (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                f"You are a strict legal/government order assistant.\n\n"
                f"Answer the user's query relying ONLY on the provided Context. "
                f"First, identify the exact G.O. number, department, date, and subject from the Context.\n"
                f"If the query asks about what the G.O. is about, to explain it, or to summarize it, summarize the complete Government Order in 3 to 6 bullet points detailing all major decisions taken. Do not quote a single section unless specifically asked for a single section.\n"
                f"If the retrieved Context is about a different G.O. or a different subject than what the user query asks about, say exactly:\n"
                f"\"The provided context does not contain the answer for this question.\"\n\n"
                f"Do not mix information from other documents. Do not answer from general knowledge or assume any missing details. "
                f"Give a short, direct answer with the source G.O. number.\n\n"
                f"Context:\n{context_str}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n{query}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        else:
            prompt = (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                f"You are a strict legal assistant. The user's query was not found in the legal act database. "
                f"State clearly and directly that you could not find any matching sections in the provided documents to answer the question, and do not provide any general legal advice or guess the answer.<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n{query}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
            
        # Run inference
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                eos_token_id=tokenizer.eos_token_id
            )
            
        # Extract and decode response text
        generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
        response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        return jsonify({
            "response": response_text,
            "context": context_docs
        })
        
    except Exception as e:
        print(f"Inference error: {e}")
        return jsonify({
            "response": f"Error generating response: {e}",
            "context": context_docs
        })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=args.port, debug=False)
