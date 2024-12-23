# This Data is taken from  Baby Llama: knowledge distillation from an ensemble of teachers trained on a small dataset with no performance penalty
import re
from pathlib import Path


DATA_ROOT = Path("./")
DATA_SPLITS = ["babylm_100M", "babylm_dev"]

# test if the data is available and path working
for split in DATA_SPLITS:
    input_dir = DATA_ROOT / "babylm_data" / split
    train_files = [
        f for f in input_dir.iterdir() if f.is_file() and f.suffix in [".train", ".dev"]
    ]
    print(f"Files in {split}:", [f.name for f in train_files])


# Help functions for the cleanup functions
def cleanup_extra_spaces(input_text):
    """Remove extra spaces and space before punctuation."""
    multiple_spaces_ex = re.compile(r"[ \t\u00A0]+")
    space_before_punctuation_ex = re.compile(r"[ \t\u00A0]([.,;!?])")
    input_text = multiple_spaces_ex.sub(" ", input_text)
    input_text = space_before_punctuation_ex.sub(r"\1", input_text)
    return input_text


# defining the clean up functions
def cleanup_simple_wikipedia(input_text):
    """Clean up Wikipedia input_text (replacing double line breaks with a single one)."""
    # input_text = re.sub(r"\n\n", "\n", input_text) # maybe not needed to keep text structure
    # maybe remove coordinates?
    input_text = cleanup_extra_spaces(input_text)
    return input_text

def cleanup_wikipedia(input_text):
    """Remove certain formatting from Wikipedia input_text."""
    input_text = re.sub(r"= = = (.+?) = = =\n", r"\1", input_text)
    lines = [line.strip() for line in input_text.splitlines()]
    input_text = cleanup_extra_spaces("\n".join(lines))
    return input_text




def cleanup_qed(input_text):
    """Clean QED input_text by removing:
    - Lines between 15,000 and 19,209.
    - Brackets and their content.
    - Uppercase, (double, triple) punctuation, and dashes.
    - All variations of &amp;gt;, &amp;amp;gt, etc.
    - Entire substrings between two whitespaces if they contain 'amp;'.
    """

    # Split the input text into lines and filter out the specified range
    lines = input_text.splitlines()
    lines = [line for idx, line in enumerate(lines, start=1) if idx < 15000 or idx > 19209]

    # Join back the filtered lines for further processing
    input_text = "\n".join(lines)

    punctuation_ex = re.compile(r"([.!?]\s*)")  # Matches punctuation followed by any whitespace
    unimportant_chars_ex = re.compile(r"\(.*?\)|\[.*?\]")  # Matches any content within () or []
    dash_ex = re.compile(r"--")  # double or single dashes
    repeated_punctuation_ex = re.compile(r"([.!?]){2,}")  # two or more of the same punctuation mark
    amp_gt_variants_ex = re.compile(r"&amp;gt;|&amp;amp;gt;?|&gt;|&amp;")  # Matches all variations of amp;gt
    amp_in_words_ex = re.compile(r"\s\S*amp;\S*\s")  # Matches substrings with 'amp;' between whitespaces

    lines = []
    for line in input_text.splitlines():
        # Remove double dashes and unimportant content
        line = dash_ex.sub("", line)  # Remove double dashes
        line = unimportant_chars_ex.sub("", line)  # Remove any content within () or []
        line = repeated_punctuation_ex.sub(r"\1", line)  # Remove repeated punctuation

        # Remove all variations of &amp;gt, &gt, etc.
        line = amp_gt_variants_ex.sub("", line)

        # Remove substrings containing 'amp;' between two whitespaces
        line = amp_in_words_ex.sub(" ", line)

        # Only proceed if there's content in line after cleanup
        if line.strip():
            f_upper = sum(c.isupper() for c in line) / len(line) if len(line) > 0 else 0
            if f_upper >= 0.5:  # Mostly uppercase characters
                split_on_punctuation = punctuation_ex.split(line.replace("l", "I"))
                # Capitalize sentences split by punctuation marks
                line = "".join(
                    [sentence.capitalize() for sentence in split_on_punctuation]
                )

            lines.append(line.strip())  # Add cleaned line to output
    cleaned_text = "\n".join(lines)
    return cleanup_extra_spaces(cleaned_text)  # apply extra space cleanup




def cleanup_bnc_spoken(input_text):
    """Clean BNC spoken data by removing extra spaces and double newlines."""
    input_text = cleanup_extra_spaces(input_text)
    # input_text = re.sub(r"\n\n", "\n", input_text)  # needed because speech
    return input_text


def cleanup_aochildes(input_text):
    """Clean AO-CHILDES input_text by removing extra spaces."""
    return cleanup_extra_spaces(input_text)


def cleanup_cbt(input_text):
    """Clean CBT input_text, fixing space around apostrophes
    Remove _BOOK_TITLE_ : Andrew_Lang___Prince_Prigio.txt.out?
    """
    input_text = cleanup_extra_spaces(input_text)

    # Regex patterns
    # space before apostrophes
    space_before_apostroph = re.compile(r"([\w\d])[ \t\u00A0](['’]\w)")
    dash_ex = re.compile(r"--")  # double or single dashes
    # backticks and single quotes
    backtick_pattern = re.compile(r"`{1,}")
    single_quote_pattern = re.compile(r"'{1,}")
    # remove whitespace inside quotes
    spaces_inside_quotes_pattern = re.compile(r'"[\s]*(.+?)[\s]*"')
    lsb_rsb_pattern = re.compile(r"-LSB-\s*\d*\s*-RSB-")

    # Apply regex patterns/substitutions
    input_text = space_before_apostroph.sub(r"\1\2", input_text)
    input_text = dash_ex.sub("", input_text)

    # replace backticks and single quotes with double quotes
    input_text = backtick_pattern.sub(r'"', input_text)
    input_text = single_quote_pattern.sub(r'"', input_text)
    # not working perfectly but fine enough -> not good with newlines and ongoing quotes
    input_text = spaces_inside_quotes_pattern.sub(r'"\1"', input_text)

    input_text = lsb_rsb_pattern.sub("", input_text)

    return input_text


def cleanup_children_stories(input_text):
    """Clean children's stories data by cleaning extra spaces."""
    # input_text = re.sub(r"\n\n", "\n", input_text) # not needed to keep text structure
    return cleanup_extra_spaces(input_text)


def cleanup_gutenberg(input_text):
    """Clean gutenberg by removing double dashes."""
    # sub with whitespace because it often appears between words
    input_text = re.sub(r"--", " ", input_text)
    input_text = re.sub("_", "", input_text)
    return input_text


def cleanup_open_subtitles(input_text):
    """Remove subtitle credits and different dashes in OpenSubtitles data
    Maybe think about removing brackets or whitespaces in brackets
    [Audience Applauding, Cheering].
    """
    subtitle_credit_ex = re.compile(r"^.*subtitle.*$\n", re.MULTILINE | re.IGNORECASE)
    line_start_dash_pattern = re.compile(r"^-+\s*", re.MULTILINE)
    # apply regex patterns/substitutions
    input_text = subtitle_credit_ex.sub("", input_text)
    input_text = line_start_dash_pattern.sub("", input_text)
    input_text = re.sub(r"--", " ", input_text)
    return input_text


def cleanup_switchboard(input_text):
    """Return Switchboard input_text as-is, exept for extra spaces."""
    return cleanup_extra_spaces(input_text)


# Mapping of cleanup functions
CLEANUP_FUNCTIONS = {
    "aochildes": cleanup_aochildes,
    "bnc_spoken": cleanup_bnc_spoken,
    "cbt": cleanup_cbt,
    "children_stories": cleanup_children_stories,
    "gutenberg": cleanup_gutenberg,
    "open_subtitles": cleanup_open_subtitles,
    "qed": cleanup_qed,
    "simple_wikipedia": cleanup_simple_wikipedia,
    "switchboard": cleanup_switchboard,
    "wikipedia": cleanup_wikipedia,
}


# Main Execution
if __name__ == "__main__":
    # Clean up the data
    for split in DATA_SPLITS:
        INPUT_DIR = DATA_ROOT / "babylm_data" / split
        OUTPUT_DIR = DATA_ROOT / "babylm_data_clean" / f"{split}_clean"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        train_files = [
            f
            for f in INPUT_DIR.iterdir()
            if f.is_file() and f.suffix in [".train", ".dev"]
        ]
        for file in train_files:
            text = file.read_text()
            cleaned_text = CLEANUP_FUNCTIONS[file.stem](text)
            (OUTPUT_DIR / file.name).write_text(cleaned_text)
            print(
                f"🧹 Cleaned '{file.name}' (size {len(text)} -> {len(cleaned_text)}) in {split}"
            )
    print("🧹 All done!")
