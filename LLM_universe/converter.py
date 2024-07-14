import os
import nbformat
from deep_translator import GoogleTranslator

# Function to split text into chunks within the allowed limit
def split_text(text, max_length=500):
    return [text[i:i + max_length] for i in range(0, len(text), max_length)]

# Function to translate text using the deep-translator package
def translate_text(text, src_lang='zh-CN', dest_lang='en'):
    translator = GoogleTranslator(source=src_lang, target=dest_lang)
    chunks = split_text(text)
    translated_chunks = []
    for chunk in chunks:
        try:
            translated_chunk = translator.translate(chunk)
            if translated_chunk is not None:
                translated_chunks.append(translated_chunk)
            else:
                print(f"Translation returned None for chunk: {chunk[:50]}...")  # Print part of the chunk
                translated_chunks.append(chunk)  # Fallback to original text if translation fails
        except Exception as e:
            print(f"Error translating text: {e}")
            translated_chunks.append(chunk)  # Fallback to original text if translation fails
    return ''.join(translated_chunks)

# Function to translate a Jupyter notebook
def translate_notebook(input_path, output_path, src_lang='zh-CN', dest_lang='en'):
    # Load the notebook
    with open(input_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # Translate each cell
    for cell in nb.cells:
        if cell.cell_type == 'markdown':
            cell.source = translate_text(cell.source, src_lang, dest_lang)
        elif cell.cell_type == 'code':
            # Optionally translate comments and docstrings in code cells
            lines = cell.source.split('\n')
            translated_lines = []
            for line in lines:
                if line.strip().startswith('#') or line.strip().startswith('"""') or line.strip().startswith("'''"):
                    translated_lines.append(translate_text(line, src_lang, dest_lang))
                else:
                    translated_lines.append(line)
            cell.source = '\n'.join(translated_lines)

    # Save the translated notebook
    with open(output_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

# Function to translate a markdown file
def translate_markdown(input_path, output_path, src_lang='zh-CN', dest_lang='en'):
    # Read the markdown file
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Translate the content
    translated_content = translate_text(content, src_lang, dest_lang)
    
    # Save the translated content
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(translated_content)

# Function to walk through directories and translate .ipynb and .md files
def translate_files_in_directory(root_dir, src_lang='zh-CN', dest_lang='en'):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.ipynb') or filename.endswith('.md'):
                input_path = os.path.join(dirpath, filename)
                # Translate the filename itself
                translated_filename = translate_text(filename, src_lang, dest_lang)
                # Remove invalid characters from translated filename
                valid_translated_filename = "".join(c if c.isalnum() or c in [' ', '.', '_'] else "_" for c in translated_filename)
                output_filename = f"translated_{valid_translated_filename}"
                output_path = os.path.join(dirpath, output_filename)
                if filename.endswith('.ipynb'):
                    translate_notebook(input_path, output_path, src_lang, dest_lang)
                elif filename.endswith('.md'):
                    translate_markdown(input_path, output_path, src_lang, dest_lang)
                print(f"Translated {input_path} to {output_path}")

# Example usage
root_directory = r'D:\NLP_works\Data_whale_cn\llm-cookbook-main'  # Path to the root directory containing notebooks and markdown files
translate_files_in_directory(root_directory, src_lang='zh-CN', dest_lang='en')
