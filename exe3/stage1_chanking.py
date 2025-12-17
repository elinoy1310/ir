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


    return child_chunks


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
    chunk_index = {}

    for filename in os.listdir(input_dir):
        if not filename.endswith(".txt"):
            continue

        file_path = os.path.join(input_dir, filename)

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunk_text_by_sentences(
            text=text,
            source_path=file_path
        )
        chunk_index[file_path] = []

        # Save each chunk as a separate file
        for chunk in chunks:
            output_file = os.path.join(output_dir, f"chunk_{chunk_id}.txt")

            with open(output_file, "w", encoding="utf-8") as out:
                out.write(chunk["text"])

            chunk_index[file_path].append(output_file)

            chunk_id += 1
    
    index_path = os.path.join(output_dir, "chunk_index.json")

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(chunk_index, f, indent=2, ensure_ascii=False)

def process_directory_for_parent_child_chunking(input_dir, output_dir):
    """
    Iterate over all .txt files in input_dir, apply parent-child chunking,
    and save child chunks to output_dir with a JSON index.
    """

    os.makedirs(output_dir, exist_ok=True)

    chunk_id = 0
    chunk_index = {}

    for filename in os.listdir(input_dir):
        if not filename.endswith(".txt"):
            continue

        file_path = os.path.join(input_dir, filename)

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        child_chunks = parent_child_chunking(
            text=text,
            source_path=file_path
        )

        for chunk in child_chunks:
            output_file = os.path.join(output_dir, f"child_chunk_{chunk_id}.txt")

            # Save only the child text
            with open(output_file, "w", encoding="utf-8") as out:
                out.write(chunk["child_text"])
            
            if file_path not in chunk_index:
                chunk_index[file_path] = [] 

            # Index entry
            chunk_index[file_path].append({
                "chunk_file": output_file,
                "parent_id": chunk["parent_id"],
                "child_id": chunk["child_id"],
                "parent_text": chunk["parent_text"]
            })

            chunk_id += 1

    # Save index
    index_path = os.path.join(output_dir, "parent_child_chunk_index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(chunk_index, f, indent=2, ensure_ascii=False)



if __name__ == "__main__":

    process_directory_for_chunking("exe3/clean-text/UK", "exe3/fixed-chunked-text/UK")
    process_directory_for_chunking("exe3/clean-text/US", "exe3/fixed-chunked-text/US")
   
    process_directory_for_parent_child_chunking("exe3/clean-text/UK", "exe3/parent-child-chunked-text/UK")
     
    process_directory_for_parent_child_chunking("exe3/clean-text/US", "exe3/parent-child-chunked-text/US")