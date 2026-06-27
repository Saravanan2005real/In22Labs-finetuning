import os
import re
import json
import glob
from markitdown import MarkItDown
from elasticsearch import Elasticsearch

def parse_sections(text, act_name):
    """
    Parses sections/articles from a legal act markdown text.
    Identifies lines like "1. Short title...", "Section 2. Definitions...", etc.
    """
    lines = text.split('\n')
    sections = []
    current_section = None
    current_content = []
    
    # Regex to match lines starting with:
    # - "1. Title"
    # - "Section 2. Title"
    # - "1A. Title"
    # Note: Avoid matching list items like "(a) text" or short bullet points.
    section_pattern = re.compile(
        r'^(?:#+\s*|\*+\s*|Section\s+|Sec\.\s*)?(\d+[A-Za-z]*)\.?\s+([A-Za-z].*)$'
    )
    
    for line in lines:
        line_stripped = line.strip()
        match = section_pattern.match(line_stripped)
        
        # Ensure we don't accidentally match minor subsections/list items like "(1) text" or "1) text"
        is_subsection = line_stripped.startswith('(') or (line_stripped.startswith(')') and len(line_stripped) > 1)
        
        if match and not is_subsection:
            # Save previous section if exists
            if current_section:
                content_text = "\n".join(current_content).strip()
                if content_text:
                    current_section['content'] = content_text
                    sections.append(current_section)
            
            sec_num = match.group(1)
            sec_title = match.group(2).strip('*# ')
            
            current_section = {
                'act_name': act_name,
                'section_num': sec_num,
                'section_title': sec_title,
                'content': ''
            }
            current_content = [line]  # Include header line in content
        else:
            if current_section is not None:
                current_content.append(line)
                
    # Add final section
    if current_section:
        content_text = "\n".join(current_content).strip()
        if content_text:
            current_section['content'] = content_text
            sections.append(current_section)
            
    # Fallback: if no sections were parsed (poor OCR or odd structure), treat the whole document as one section
    if not sections:
        sections.append({
            'act_name': act_name,
            'section_num': 'Full Text',
            'section_title': 'General Content',
            'content': text
        })
        
    return sections

def clean_act_name(filename):
    """
    Cleans filename to extract a readable Act Name.
    E.g., "the_indian_partnership_act_1932.pdf" -> "The Indian Partnership Act 1932"
    """
    name = os.path.splitext(filename)[0]
    name = name.replace('_', ' ').replace('-', ' ')
    # Capitalize words
    name = ' '.join(word.capitalize() for word in name.split())
    return name

def main():
    pdf_dir = "Act_Elastic search"
    pdf_files = list(set(glob.glob(os.path.join(pdf_dir, "*.pdf")) + glob.glob(os.path.join(pdf_dir, "*.PDF"))))
    
    print(f"Found {len(pdf_files)} PDF files to process.")
    
    # Initialize MarkItDown
    print("Initializing MarkItDown...")
    md = MarkItDown()
    
    # Initialize Elasticsearch client
    es_url = os.environ.get("ELASTICSEARCH_URL", "http://127.0.0.1:9200")
    print(f"Connecting to Elasticsearch ({es_url})...")
    # Using basic connection (local development, security disabled)
    es = Elasticsearch(es_url, request_timeout=30)
    
    index_name = "legal_acts"
    try:
        if es.indices.exists(index=index_name):
            print(f"Index '{index_name}' already exists. Recreating it...")
            es.indices.delete(index=index_name)
            
        # Create index with appropriate mappings
        es.indices.create(
            index=index_name,
            body={
                "mappings": {
                    "properties": {
                        "act_name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                        "section_num": {"type": "keyword"},
                        "section_title": {"type": "text"},
                        "content": {"type": "text"}
                    }
                }
            }
        )
        print(f"Index '{index_name}' created successfully.")
    except Exception as e:
        print(f"WARNING: Could not connect to or configure Elasticsearch: {e}")
        print("We will skip Elasticsearch indexing for now, but will continue generating the training dataset.")
        es = None

    all_sections = []
    training_data = []
    
    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        act_name = clean_act_name(filename)
        print(f"\nProcessing {filename} -> '{act_name}'...")
        
        try:
            # Convert PDF to markdown
            result = md.convert(pdf_path)
            markdown_content = result.text_content
            
            # Parse sections
            sections = parse_sections(markdown_content, act_name)
            print(f"Extracted {len(sections)} sections/articles.")
            
            for sec in sections:
                all_sections.append(sec)
                
                # Index into Elasticsearch if available
                if es:
                    try:
                        es.index(index=index_name, document=sec)
                    except Exception as ex:
                        print(f"Error indexing section {sec['section_num']} of {act_name}: {ex}")
                
                # Generate training QA pairs
                # Pair 1: Direct explanation
                training_data.append({
                    "instruction": f"What is Section {sec['section_num']} of the {sec['act_name']} about, and what does it state?",
                    "output": f"Section {sec['section_num']} of the {sec['act_name']} is titled '{sec['section_title']}'. It states:\n\n{sec['content']}"
                })
                # Pair 2: Explaining rules/regulations
                training_data.append({
                    "instruction": f"Explain the legal provisions and details in Section {sec['section_num']} ({sec['section_title']}) of the {sec['act_name']}.",
                    "output": f"According to Section {sec['section_num']} of the {sec['act_name']}, which deals with '{sec['section_title']}':\n\n{sec['content']}"
                })
                # Pair 3: Quoting text
                training_data.append({
                    "instruction": f"Provide the official text of Section {sec['section_num']} under the {sec['act_name']}.",
                    "output": f"The legal text of Section {sec['section_num']} ('{sec['section_title']}') of the {sec['act_name']} is as follows:\n\n{sec['content']}"
                })
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            
    # Save the dataset to a JSON file
    dataset_path = "dataset.json"
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(training_data, f, indent=4, ensure_ascii=False)
        
    print(f"\nDataset generation complete! Created {len(training_data)} QA samples.")
    print(f"Saved training dataset to {dataset_path}")

if __name__ == "__main__":
    main()
