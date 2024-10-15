import os
import glob
import tiktoken

def count_tokens(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        content = file.read()
        tiktoken_model = tiktoken.encoding_for_model('text-embedding-3-small')
        return len(tiktoken_model.encode(content, disallowed_special=()))

def average_token_count(directory):
    text_files = glob.glob(os.path.join(directory, '*.txt'))
    if not text_files:
        return 0
    
    total_tokens = sum(count_tokens(file) for file in text_files)
    return total_tokens / len(text_files)

def main():
    root_directory = 'document_cache/vidore'
    
    for subdir in os.listdir(root_directory):
        subdir_path = os.path.join(root_directory, subdir)
        if os.path.isdir(subdir_path):
            avg_tokens = average_token_count(subdir_path)
            print(f"{subdir}: {avg_tokens:.2f} tokens on average")

if __name__ == "__main__":
    main()
