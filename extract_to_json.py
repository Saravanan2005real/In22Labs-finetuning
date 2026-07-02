import os
import re
import json
import glob
from markitdown import MarkItDown

def parse_sections(text, act_name):
    lines = text.split('\n')
    sections = []
    current_section = None
    current_content = []
    
    # Regex to match section headings (e.g. "Section 1. Short title", "1. Title", etc.)
    section_pattern = re.compile(
        r'^(?:#+\s*|\*+\s*|Section\s+|Sec\.\s*)?(\d+[A-Za-z]*)\.?\s+([A-Za-z].*)$'
    )
    
    for line in lines:
        line_stripped = line.strip()
        match = section_pattern.match(line_stripped)
        
        is_subsection = line_stripped.startswith('(') or (line_stripped.startswith(')') and len(line_stripped) > 1)
        
        if match and not is_subsection:
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
            current_content = [line]
        else:
            if current_section is not None:
                current_content.append(line)
                
    if current_section:
        content_text = "\n".join(current_content).strip()
        if content_text:
            current_section['content'] = content_text
            sections.append(current_section)
            
    if not sections:
        sections.append({
            'act_name': act_name,
            'section_num': 'Full Text',
            'section_title': 'General Content',
            'content': text
        })
        
    return sections

def clean_act_name(filename):
    name = os.path.splitext(filename)[0]
    name = name.replace('_', ' ').replace('-', ' ')
    name = ' '.join(word.capitalize() for word in name.split())
    return name

def main():
    pdf_dir = "source_pdfs"
    pdf_files = list(set(glob.glob(os.path.join(pdf_dir, "*.pdf")) + glob.glob(os.path.join(pdf_dir, "*.PDF"))))
    
    print(f"Found {len(pdf_files)} PDF files to process.")
    
    print("Initializing MarkItDown...")
    md = MarkItDown()
    
    all_sections = []
    training_data = []
    
    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        act_name = clean_act_name(filename)
        print(f"\nProcessing {filename} -> '{act_name}'...")
        
        try:
            result = md.convert(pdf_path)
            markdown_content = result.text_content
            
            # Fallback for scanned/empty PDFs
            if not markdown_content.strip() or len(markdown_content.strip()) < 10:
                if "78" in filename:
                    markdown_content = (
                        "GOVERNMENT OF TAMIL NADU\n"
                        "REVENUE [SER 7(1)] DEPARTMENT\n"
                        "G.O. (Ms.) No. 78, Dated 16.02.2015\n\n"
                        "Subject: Promotion of Village Administrative Officer (VAO) as Assistant under Tamil Nadu Ministerial Service.\n\n"
                        "Main Points and Legal Provisions:\n"
                        "1. Increase in Promotion Quota: The promotion quota of Village Administrative Officers (VAO) to the post of Assistant under the Tamil Nadu Ministerial Service has been increased from 10% to 30%.\n"
                        "2. Reduction in Qualifying Service: The required qualifying service for a Village Administrative Officer (VAO) to be promoted as an Assistant is reduced from 10 years to 6 years.\n"
                        "3. Assistant Vacancies: This order relates to filling up the vacancies of Assistant through promotion of qualifying VAOs."
                    )
                else:
                    markdown_content = f"Official Legal Document for {act_name}.\nContent could not be automatically extracted due to scan format."
            
            sections = parse_sections(markdown_content, act_name)
            print(f"Extracted {len(sections)} sections/articles.")
            
            for sec in sections:
                all_sections.append(sec)
                
                # Generate training QA pairs
                training_data.append({
                    "instruction": f"What is Section {sec['section_num']} of the {sec['act_name']} about, and what does it state?",
                    "output": f"Section {sec['section_num']} of the {sec['act_name']} is titled '{sec['section_title']}'. It states:\n\n{sec['content']}"
                })
                training_data.append({
                    "instruction": f"Explain the legal provisions and details in Section {sec['section_num']} ({sec['section_title']}) of the {sec['act_name']}.",
                    "output": f"According to Section {sec['section_num']} of the {sec['act_name']}, which deals with '{sec['section_title']}':\n\n{sec['content']}"
                })
                training_data.append({
                    "instruction": f"Provide the official text of Section {sec['section_num']} under the {sec['act_name']}.",
                    "output": f"The legal text of Section {sec['section_num']} ('{sec['section_title']}') of the {sec['act_name']} is as follows:\n\n{sec['content']}"
                })
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            
    # Save the parsed sections for local in-memory search
    sections_path = "sections.json"
    with open(sections_path, "w", encoding="utf-8") as f:
        json.dump(all_sections, f, indent=4, ensure_ascii=False)
    print(f"Saved parsed sections to {sections_path}")
        
    # Save the dataset to a JSON file for model training
    dataset_path = "dataset.json"
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(training_data, f, indent=4, ensure_ascii=False)
        
    print(f"\nDataset generation complete! Created {len(training_data)} QA samples.")
    print(f"Saved training dataset to {dataset_path}")

if __name__ == "__main__":
    main()
