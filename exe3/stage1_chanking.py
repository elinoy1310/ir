# exe3/stage1_chanking.py
import os
import re
import json

import re

def parent_child_chunking(text,source_path,parent_max_words=1200,child_max_words=330):
    """
    Split text into parent-child chunks.
    Parent chunks provide broad context.
    Child chunks are smaller and used for retrieval.

    Returns a list of child chunks, each including its parent context.
    """
    # Step 1: Create parent chunks
    parents_chunks = chunk_text_by_sentences(text, source_path, parent_max_words, overlap_sentences=1)
    parents_chunks = [p["text"] for p in parents_chunks]
    print(f"Created {len(parents_chunks)} parent chunks for {source_path}")
    # Step 2: Build child chunks inside each parent
    child_chunks = []

    for parent_id, parent_text in enumerate(parents_chunks):

        children_chunks=chunk_text_by_sentences(parent_text,source_path,child_max_words,overlap_sentences=2)

        for child_id,child in enumerate(children_chunks):
            child_chunks.append({
                "source_path": source_path,
                "parent_id": parent_id,
                "child_id": child_id,
                "parent_text": parent_text,
                "child_text": child["text"]
            })


    return parents_chunks,child_chunks


def chunk_text_by_sentences(text, source_path, max_words=660, overlap_sentences=3):
    """
    Split text into overlapping chunks of full sentences.
    Each chunk has up to max_words words and overlaps with
    the previous chunk by overlap_sentences sentences.
    """

    # Split text into sentences (simple but robust enough for cleaned text)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_word_count = 0
    i = 0
    problem=False

    while i < len(sentences):
        sentence = sentences[i]
        sentence_word_count = len(sentence.split())

        if sentence_word_count > max_words or problem:
            problem=False
            print(f"Warning: A single sentence exceeds the max word limit of {max_words} words. Sentence will be skipped.")
            print(f"Sentence {i} in {source_path}")
            print(f"the sentence: {sentence[:50]}... {sentence_word_count} words")
  
            sub_sentences = re.split(r'(?<=,)\s+', sentence)
            if len(sub_sentences) >1:
                print("Splitting the sentence into smaller sub-sentences. size:", len(sub_sentences))
                print("before split len(sentences)):", len(sentences))
                # print(type(sub_sentences))
            
                sentences =sentences[:i] + sub_sentences+ sentences[i+1:]
                print("after split len(sentences)):", len(sentences))
                continue
            else:
                print("Skipping sentence as it cannot be split further.")
                print(len(sub_sentences))
                i += 1
                continue
  
        # If adding this sentence exceeds max_words, finalize current chunk
        if current_word_count + sentence_word_count > max_words and current_chunk:
            chunks.append({
                "source_path": source_path,
                "text": " ".join(current_chunk)
            })

            # Start new chunk with overlap
            
            current_chunk = current_chunk[-overlap_sentences:] if overlap_sentences > 0 and overlap_sentences < len(current_chunk) else []

            current_word_count = sum(len(s.split()) for s in current_chunk)
            if current_word_count + sentence_word_count > max_words and current_chunk:
                print("problem")
                problem=True

        else:
            current_chunk.append(sentence)
            current_word_count += sentence_word_count
            i += 1

    # Add the last chunk
    if current_chunk:
        chunks.append({
            "source_path": source_path,
            "text": " ".join(current_chunk)
        })

    return chunks


def process_directory_for_chunking(input_dir, output_dir):
    """
    Iterate over all .txt files in input_dir, chunk them,
    and save the results into output_dir.
    """

    os.makedirs(output_dir, exist_ok=True)

    chunk_id = 0
    chunk_index = {}  # מפתח: קובץ מקור -> רשימת צ'אנקים
    reverse_chunk_index = {}  # מפתח: צ'אנק -> קובץ מקור

    for filename in os.listdir(input_dir):
        if not filename.endswith(".txt"):
            continue

        file_path = input_dir+"/"+filename

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunk_text_by_sentences(
            text=text,
            source_path=file_path
        )
        chunk_index[file_path] = []

        # Save each chunk as a separate file
        for chunk in chunks:
            output_file = output_dir+f"/chunk_{chunk_id}.txt"

            with open(output_file, "w", encoding="utf-8") as out:
                out.write(chunk["text"])

            chunk_index[file_path].append(output_file)
            reverse_chunk_index[output_file] = file_path  # הצ'אנק כמפתח והקובץ המקורי כערך

            chunk_id += 1
    
    # שמירת קובץ JSON של הצ'אנקים לכל קובץ מקור
    index_path = os.path.join(output_dir, "chunk_index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(chunk_index, f, indent=2, ensure_ascii=False)

    # שמירת קובץ JSON שבו כל צ'אנק הוא מפתח והקובץ המקורי הוא הערך
    reverse_index_path = os.path.join(output_dir, "reverse_chunk_index.json")
    with open(reverse_index_path, "w", encoding="utf-8") as f:
        json.dump(reverse_chunk_index, f, indent=2, ensure_ascii=False)


def process_directory_for_parent_child_chunking(input_dir, output_dir):
    """
    Iterate over all .txt files in input_dir, apply parent-child chunking,
    and save child chunks to output_dir with a JSON index.
    """

    os.makedirs(os.path.join(output_dir, 'parent'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'children'), exist_ok=True)

    p_chunk_id = 0
    c_chunk_id = 0
    start_parent_index=0
    parent_index = {}  # מפתח: קובץ מקור -> נתיב של אב
    child_index = {}  # מפתח: קובץ ילד -> [נתיב אב, נתיב מקור]

    for filename in os.listdir(input_dir):
        if not filename.endswith(".txt"):
            continue

        file_path = os.path.join(input_dir, filename)

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        parent_chunks, child_chunks = parent_child_chunking(text=text, source_path=file_path)

        
        for chunk in parent_chunks:
            parent_file = os.path.join(output_dir, 'parent', f"parent_chunk_{p_chunk_id}.txt")
            with open(parent_file, "w", encoding="utf-8") as parent_out:
                parent_out.write(chunk)
            # Update parent index with the file path of the parent chunk
            if parent_file not in parent_index:
                parent_index[parent_file] = file_path
            p_chunk_id += 1
        

        # Save parent chunks and update index
        for chunk in child_chunks:
            parent_file=os.path.join(output_dir, 'parent', f"parent_chunk_{start_parent_index + chunk['parent_id']}.txt")
            child_file = os.path.join(output_dir, 'children', f"child_chunk_{c_chunk_id}.txt")
            # Save child chunk
            with open(child_file, "w", encoding="utf-8") as child_out:
                child_out.write(chunk["child_text"])

            # Update child index: each child chunk links to the parent and the original file
            child_index[child_file] = {
                "parent_file": parent_file,
                "original_file": file_path
            }

            c_chunk_id += 1
        start_parent_index=p_chunk_id

    # Save the parent-to-original file mapping in a JSON
    parent_index_path = os.path.join(output_dir, 'parent', "parent_index.json")
    with open(parent_index_path, "w", encoding="utf-8") as f:
        json.dump(parent_index, f, indent=2, ensure_ascii=False)

    # Save the child-to-parent and original file mapping in a JSON
    child_index_path = os.path.join(output_dir, 'children', "child_index.json")
    with open(child_index_path, "w", encoding="utf-8") as f:
        json.dump(child_index, f, indent=2, ensure_ascii=False)



if __name__ == "__main__":
    # process_directory_for_chunking("exe3/clean-text/UK", "exe3/fixed-chunked-text/UK")
    # process_directory_for_chunking("exe3/clean-text/UK", "exe3/fixed-chunked-text/UK")
    # process_directory_for_chunking("exe3/clean-text/UK", "exe3/chunked-text/UK")
    # process_directory_for_chunking("exe3/clean-text/US", "exe3/chunked-text/US")
    ##process_directory_for_parent_child_chunking("exe3", "exe3/chunked-temp")
    process_directory_for_parent_child_chunking("exe3/clean-text/UK", "exe3/parent-child-chunked-text/UK")
    process_directory_for_parent_child_chunking("exe3/clean-text/US", "exe3/parent-child-chunked-text/US")