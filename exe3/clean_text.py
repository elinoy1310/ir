# exe3/clean_text.py
import os
import re
import html

def clean_text(text, remove_header=False):
    text = html.unescape(text)
    text = text.replace('”', '"').replace('“', '"').replace('’', "'").replace('‘', "'").replace('``', '"').replace("`", "'")
    text = re.sub(r'[^\x00-\x7F\n\t\s]+', '', text)
    text = re.sub(r'([.!?])(?=\S)', r'\1', text)
    text = re.sub(r'</?pre>', '', text)
    if remove_header:
        text = re.sub(r'Title:.*?(\d{4}).*?\1', '', text, flags=re.DOTALL)
    text = re.sub(r'_+', '', text)
    text = re.sub(r'\[\[Page.*?\]\]', '', text)
    text = text.strip()
    # Replace multiple spaces with a single space
    text = re.sub(r'\s\s+', '\n', text)
    return text

def process_files_in_directory(input_directory, output_directory, remove_header=False):
    # Ensure the output directory exists, create it if it doesn't
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Iterate through all files in the input directory
    for filename in os.listdir(input_directory):
        file_path = os.path.join(input_directory, filename)
        
        # Process only .txt files
        if os.path.isfile(file_path) and file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()

            # Clean the text
            cleaned_text = clean_text(text, remove_header)

            # Define the path for the cleaned text file in the output directory
            output_file_path = os.path.join(output_directory, filename)

            # Save the cleaned text to the output file
            with open(output_file_path, 'w', encoding='utf-8') as file:
                file.write(cleaned_text)
    print(f'Processed and saved cleaned text in {output_directory}')

if __name__ == '__main__':
    process_files_in_directory("exe3/UK", "exe3/clean-text/UK")
    process_files_in_directory("exe3/US", "exe3/clean-text/US", remove_header=True)
