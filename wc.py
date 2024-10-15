import os
import glob

def count_words(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        content = file.read()
        words = content.split()
        return len(words)

def average_word_count(directory):
    text_files = glob.glob(os.path.join(directory, '*.txt'))
    if not text_files:
        return 0
    
    total_words = sum(count_words(file) for file in text_files)
    return total_words / len(text_files)

def main():
    root_directory = 'document_cache/vidore'
    
    for subdir in os.listdir(root_directory):
        subdir_path = os.path.join(root_directory, subdir)
        if os.path.isdir(subdir_path):
            avg_words = average_word_count(subdir_path)
            print(f"{subdir}: {avg_words:.2f} words on average")

if __name__ == "__main__":
    main()
