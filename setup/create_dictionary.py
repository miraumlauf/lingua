import json
import re
from pathlib import Path
from nltk.tokenize import sent_tokenize
from lingua.tokenizer import build_tokenizer

DATA_ROOT = Path("./")
DATA_SPLITS = ["babylm_100M_clean", "babylm_dev_clean"] # parent dir is also babylm_data_clean

# test if the data is available and path working
for split in DATA_SPLITS:
    input_dir = DATA_ROOT / "babylm_data_clean" / split
    train_files = [
        f for f in input_dir.iterdir() if f.is_file() and f.suffix in [".train", ".dev"]
    ]
    print(f"Files in {split}:", [f.name for f in train_files])

SEQ_LEN = 512  # Maximum number of tokens per JSON entry
TOKENIZER_PATH = "./tokenizers/llama3/original/tokenizer.model"
TOKENIZER = build_tokenizer(name="tiktoken", path=TOKENIZER_PATH)



def split_conversational(text, seq_len=SEQ_LEN, tokenizer=TOKENIZER):
    # change name of function -> all occurences
    """
    Splits text into JSON entries where each entry is a single JSON object
    formatted for line-separated output.
    """    
    sentences = sent_tokenize(text)
    buffer = ""  # Temporary buffer to accumulate sentences

    for sentence in sentences:
        # Combine the current buffer with the next sentence
        temp_buffer = f"{buffer} {sentence}".strip()
        
        # Count the number of tokens in the combined text
        token_count = len(tokenizer.encode(temp_buffer, add_bos=False, add_eos=False))
        
        # If the combined text exceeds the token limit, yield the buffer as a JSON object
        if token_count > seq_len:
            if buffer:  # Ensure the buffer is not empty
                yield json.dumps({
                    "text": buffer.strip(),
                    "token_count": len(tokenizer.encode(buffer.strip(), add_bos=False, add_eos=False))
                })  # Include token count in JSON
            buffer = sentence[:seq_len]  # Start a new buffer with the current sentence (truncated if too long)
        else:
            buffer = temp_buffer  # Otherwise, keep accumulating sentences

    # Yield the remaining buffer as a final JSON object if it contains text
    if buffer.strip():
        yield json.dumps({
            "text": buffer.strip(),
            "token_count": len(tokenizer.encode(buffer.strip(), add_bos=False, add_eos=False))
        })



# Defining the split up functions

# CONVERSATIONAL
def split_aochildes(input_text):
    """Conversational Data -> not double newline"""
    # print("Processing AOChildes")
    return split_conversational(input_text)
    # first_2000_lines = "\n".join(input_text.splitlines()[:2000])
    # return split_conversational(first_2000_lines)


def split_bnc_spoken(input_text):
    """Conversational Data -> not double newline (filter maybe LCB out)"""
    # print("Processing BNC Spoken")
    return split_conversational(input_text)
    # first_2000_lines = "\n".join(input_text.splitlines()[:2000])
    # return split_conversational(first_2000_lines)


def split_switchboard(input_text):
    """Also conversational"""
    # print("Processing Switchboard")
    return split_conversational(input_text)
    # first_2000_lines = "\n".join(input_text.splitlines()[:2000])
    # return split_conversational(first_2000_lines)


def split_qed(input_text):
    """ I think also conversational"""
    # print("Processing QED")
    return split_conversational(input_text)
    # first_2000_lines = "\n".join(input_text.splitlines()[:2000])
    # return split_conversational(first_2000_lines)


def split_open_subtitles(input_text):
    """Split on double newline and then SEQ Lenght"""
    return split_conversational(input_text)
    # first_2000_lines = "\n".join(input_text.splitlines()[:2000])
    # return split_conversational(first_2000_lines)


# OTHER SPLITTING MECHANISMS 

# CBT (chapters of books are split)
def split_by_chapters(input_text, seq_len=SEQ_LEN, tokenizer=TOKENIZER):
    """
    Yields:
        dict: JSON-formatted chapter or chunk with "text", "token_count", "chapter_id", and "book_title".
    """
    if tokenizer is None:
        raise ValueError("Tokenizer is not initialized.")

    books = input_text.split("_BOOK_TITLE_ : ")
    for book in books:
        if not book.strip():
            continue  # Skip empty entries
        
        # Extract the book title and its contents
        try:
            book_title, book_content = book.split("\n", 1)
        except ValueError:
            continue  # Skip malformed book entries
        
        # Split the book into chapters by "CHAPTER" markers
        chapters = book_content.split("CHAPTER")
        for chapter_id, chapter in enumerate(chapters):
            if not chapter.strip():
                continue  # Skip empty chapters
            
            buffer = ""
            for paragraph in chapter.split("\n\n"):  # Split chapter into paragraphs
                temp_buffer = f"{buffer} {paragraph}".strip()
                token_count = len(tokenizer.encode(temp_buffer, add_bos=False, add_eos=False))

                # If the buffer exceeds the sequence length, yield the current chunk
                if token_count > seq_len:
                    if buffer:
                        yield {
                            "text": buffer.strip(),
                            "token_count": len(tokenizer.encode(buffer.strip(), add_bos=False, add_eos=False)),
                            "chapter_id": chapter_id,
                            "book_title": book_title.strip(),
                        }
                    buffer = paragraph[:seq_len]  # Start a new buffer with the current paragraph
                else:
                    buffer = temp_buffer

            # Yield any remaining buffer at the end of the chapter
            if buffer.strip():
                yield {
                    "text": buffer.strip(),
                    "token_count": len(tokenizer.encode(buffer.strip(), add_bos=False, add_eos=False)),
                    "chapter_id": chapter_id,
                    "book_title": book_title.strip(),
                }


def split_cbt(input_text):
    """Split by __BOOKTITLE__ and then CHAPTER"""
    return split_by_chapters(input_text)
    # first_lines = "\n".join(input_text.splitlines()[:7000])
    # return split_by_chapters(first_lines)


# CHILDREN STORIES
def split_newline_text(input_text, seq_len=SEQ_LEN, tokenizer=TOKENIZER):
    """
    Function to split children's stories into JSON entries based on sequence length.
    Only outputs "text" and "token_count" keys.
    """

    def process_buffer(buffer):
        """Helper to split buffer into chunks by sequence length and yield JSON entries."""
        sentences = sent_tokenize(buffer.strip())
        chunk = ""

        for sentence in sentences:
            temp_chunk = f"{chunk} {sentence}".strip()
            token_count = len(tokenizer.encode(temp_chunk, add_bos=False, add_eos=False))

            if token_count > seq_len:
                if chunk.strip():
                    yield {"text": chunk.strip(), "token_count": len(tokenizer.encode(chunk.strip(), add_bos=False, add_eos=False))}
                chunk = sentence
            else:
                chunk = temp_chunk

        if chunk.strip():
            yield {"text": chunk.strip(), "token_count": len(tokenizer.encode(chunk.strip(), add_bos=False, add_eos=False))}

    stories = input_text.split("\n\n")  # Split by double newline
    buffer = ""

    for part in stories:
        if part.strip():
            buffer = f"{buffer} {part.strip()}".strip()
            token_count = len(tokenizer.encode(buffer, add_bos=False, add_eos=False))

            if token_count > seq_len:
                yield from process_buffer(buffer)
                buffer = ""  # Reset the buffer

    if buffer.strip():  # Process the remaining text
        yield from process_buffer(buffer)


def split_children_stories(input_text):
    """can be split by double newline PLUS Seq lenght"""
    return split_newline_text(input_text)
    # first_lines = "\n".join(input_text.splitlines()[:6000])
    # return split_newline_text(first_lines)
    

# WIKIPEDIA and GUTENBERG

def split_paragraph_data(text, seq_len=SEQ_LEN, tokenizer=TOKENIZER):
    """
    Splits paragraph data into chunks at double newlines and ensures each chunk
    does not exceed the specified sequence length.
    """
    # Split the text into sections at two or more consecutive newlines
    sections = re.split(r"\n{2,}", text)
    buffer = ""

    for section in sections:
        temp_buffer = f"{buffer} {section}".strip()
        token_count = len(tokenizer.encode(temp_buffer, add_bos=False, add_eos=False))

        if token_count > seq_len:
            # If the buffer is not empty, yield the current buffer
            if buffer.strip():
                yield {
                    "text": buffer.strip(),
                    "token_count": len(tokenizer.encode(buffer.strip(), add_bos=False, add_eos=False)),
                }
            # Reset the buffer to the current section
            buffer = section

            # If the section itself exceeds the seq_len, split further at sentence boundaries
            if len(tokenizer.encode(buffer, add_bos=False, add_eos=False)) > seq_len:
                sentences = sent_tokenize(buffer)
                sub_buffer = ""

                for sentence in sentences:
                    temp_sub_buffer = f"{sub_buffer} {sentence}".strip()
                    if len(tokenizer.encode(temp_sub_buffer, add_bos=False, add_eos=False)) > seq_len:
                        if sub_buffer.strip():
                            yield {
                                "text": sub_buffer.strip(),
                                "token_count": len(tokenizer.encode(sub_buffer.strip(), add_bos=False, add_eos=False)),
                            }
                        sub_buffer = sentence
                    else:
                        sub_buffer = temp_sub_buffer

                # Yield any remaining sub-buffer
                if sub_buffer.strip():
                    yield {
                        "text": sub_buffer.strip(),
                        "token_count": len(tokenizer.encode(sub_buffer.strip(), add_bos=False, add_eos=False)),
                    }

            buffer = ""
        else:
            buffer = temp_buffer

    # Yield any remaining buffer
    if buffer.strip():
        yield {
            "text": buffer.strip(),
            "token_count": len(tokenizer.encode(buffer.strip(), add_bos=False, add_eos=False)),
        }

def split_simple_wikipedia(input_text):
    """Like Gutenberg, split on double newline and then add UNTIL SEQ LENGHT"""
    return split_paragraph_data(input_text)
    # first_lines = "\n".join(input_text.splitlines()[:7000])
    # return split_paragraph_data(first_lines)


def split_wikipedia(input_text):
    """Split and newline and eventual split if entries are to long"""
    return split_paragraph_data(input_text)
    # first_lines = "\n".join(input_text.splitlines()[:7000])
    # return split_paragraph_data(first_lines)


# GUTENBERG 

def split_gutenberg(input_text):
    """Split by double newline until SEQ LEnght"""
    return split_paragraph_data(input_text)
    # first_lines = "\n".join(input_text.splitlines()[:7000])
    # return split_paragraph_data(first_lines)



SPLIT_FUNCTIONS = {
    "aochildes": split_aochildes,
    "bnc_spoken": split_bnc_spoken,
    "cbt": split_cbt,
    "children_stories": split_children_stories,
    "gutenberg": split_gutenberg,
    "open_subtitles": split_open_subtitles,
    "qed": split_qed,
    "simple_wikipedia": split_simple_wikipedia,
    "switchboard": split_switchboard,
    "wikipedia": split_wikipedia,
}


# # Dont for get to convert the files to json format

# Main Execution
if __name__ == "__main__":
    total_token_count = 0  # To store total tokens across all files

    for split in DATA_SPLITS:
        INPUT_DIR = DATA_ROOT / "babylm_data_clean" / split
        OUTPUT_DIR = DATA_ROOT / "babylm_data_json" / f"{split}_json"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        train_files = [
            f
            for f in INPUT_DIR.iterdir()
            if f.is_file() and f.suffix in [".train", ".dev"]
        ]
        for file in train_files:
            text = file.read_text()
            split_function = SPLIT_FUNCTIONS.get(file.stem)
            if split_function:
                # Collect all JSON entries as strings, one per line
                # split_text = "\n".join(split_function(text))
                #split_text = "\n".join(json.dumps(entry) for entry in split_function(text))
                split_text = "\n".join(entry if isinstance(entry, str) else json.dumps(entry) for entry in split_function(text))

                # file extension set to jsonl and write into output files
                output_file = OUTPUT_DIR / f"{file.stem}.jsonl"
                output_file.write_text(split_text)
                print(f"✔ Processed '{file.name}' -> '{output_file}'")
                
                # Token count summation
                for line in split_text.splitlines():
                    try:
                        json_entry = json.loads(line)
                        total_token_count += json_entry.get("token_count", 0)
                    except json.JSONDecodeError:
                        print(f"❌ Skipping invalid JSON entry in {output_file}")
            else:
                print(f"❌ No split function found for '{file.stem}'")
    
    print(f"🧹 All done! Total tokens across all files: {total_token_count}")
