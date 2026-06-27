import os
import sys
from datasets import Dataset
import torch
import argparse
from flask import Flask, render_template, request, jsonify
from elasticsearch import Elasticsearch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

app = Flask(__name__)

# Parse command line arguments
parser = argparse.ArgumentParser(description="Legal RAG Chatbot Server")
parser.add_argument("--mock", action="store_true", help="Run with mock LLM responses for quick UI testing")
parser.add_argument("--port", type=int, default=5000, help="Server port number")
args, unknown = parser.parse_known_args()

# Check environment variable for mock mode as well
if os.environ.get("MOCK_MODE", "").lower() in ("true", "1", "yes"):
    args.mock = True

# Global variables for model, tokenizer, and Elasticsearch
model = None
tokenizer = None
es_client = None
device = "cpu"

es_url = os.environ.get("ELASTICSEARCH_URL", "http://127.0.0.1:9200")
print(f"Connecting to Elasticsearch ({es_url})...")
try:
    es_client = Elasticsearch(es_url, request_timeout=5)
    if es_client.ping():
        print("Successfully connected to Elasticsearch!")
    else:
        print("WARNING: Elasticsearch ping failed. Server may be starting or unreachable.")
except Exception as e:
    print(f"WARNING: Could not establish connection to Elasticsearch: {e}")

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

def search_legal_acts(query, limit=3):
    """
    Search legal acts index in Elasticsearch for relevant sections.
    """
    if not es_client:
        return []
    try:
        index_name = "legal_acts"
        if not es_client.indices.exists(index=index_name):
            return []
        
        response = es_client.search(
            index=index_name,
            body={
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["content^2", "section_title", "act_name"]
                    }
                },
                "size": limit
            }
        )
        
        hits = response['hits']['hits']
        results = []
        for hit in hits:
            results.append(hit['_source'])
        return results
    except Exception as e:
        print(f"Search error: {e}")
        return []

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/status")
def status():
    """
    Returns the current status of Elasticsearch and LLM Engine.
    """
    es_online = False
    if es_client:
        try:
            es_online = es_client.ping()
        except Exception:
            pass
            
    return jsonify({
        "elasticsearch": es_online,
        "model": args.mock or (model is not None),
        "device": "MOCK" if args.mock else device
    })

@app.route("/api/acts")
def get_acts():
    """
    Returns unique acts indexed in Elasticsearch and their document counts.
    """
    if not es_client:
        return jsonify([])
    
    try:
        index_name = "legal_acts"
        if not es_client.indices.exists(index=index_name):
            return jsonify([])
            
        # Perform aggregation to get counts per act_name
        response = es_client.search(
            index=index_name,
            body={
                "size": 0,
                "aggs": {
                    "acts": {
                        "terms": {
                            "field": "act_name.keyword",
                            "size": 50
                        }
                    }
                }
            }
        )
        
        buckets = response['aggregations']['acts']['buckets']
        acts = [{"name": bucket['key'], "count": bucket['doc_count']} for bucket in buckets]
        
        # Fallback to hardcoded list if empty for visual demo
        if not acts:
            acts = [
                {"name": "Indian Partnership Act 1932", "count": 74},
                {"name": "Powers Of Attorney Act 1882", "count": 6},
                {"name": "Indian Christian Marriage Act 1872", "count": 88}
            ]
        return jsonify(acts)
    except Exception as e:
        print(f"Error fetching acts list: {e}")
        return jsonify([])

@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Chat endpoint.
    Retrieves context from Elasticsearch, builds prompt, and runs LLM inference.
    """
    data = request.json or {}
    query = data.get("query", "").strip()
    
    if not query:
        return jsonify({"response": "Please specify a query.", "context": []})
    
    # 1. Search relevant documents in Elasticsearch
    context_docs = search_legal_acts(query)
    
    # 2. Formulate RAG response
    if args.mock:
        # Clean, direct mock response generation
        if context_docs:
            doc = context_docs[0]
            response = f"According to Section {doc['section_num']} ('{doc['section_title']}') of the {doc['act_name']}:\n\n{doc['content']}"
        else:
            response = f"I searched the indexed legal acts but could not find a specific section matching your query '{query}'. Please verify the name of the act or section you are referring to."
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
                f"You are a helpful legal assistant. Answer the user's query by summarizing and explaining the provided Context in a clear, conversational manner. "
                f"Explain the procedures and requirements mentioned in the Context (such as the need for a written notice and the details it must contain). "
                f"Rely ONLY on the facts directly stated in the Context: do not invent any legal details, requirements, real-world analogies, or examples not present in the provided text. Cite the Act name and Section number.\n\n"
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
