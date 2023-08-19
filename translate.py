import sys
import re
import argparse
import docx
import openai
from transformers import GPT2TokenizerFast

# Specify command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "input",
    help="The path to the file to translate",
    type=str,
)
parser.add_argument(
    "--model",
    help="The name of the GPT model to use for translation",
    type=str,
    default="gpt-4",
)
parser.add_argument(
    "--source_language",
    help="The language to translate from",
    type=str,
    default="Russian",
)
parser.add_argument(
    "--target_language",
    help="The language to translate to",
    type=str,
    default="English",
)
parser.add_argument(
    "--max_tokens",
    help="The maximum number of tokens to use for each translation",
    type=int,
    default=4096,
)
parser.add_argument(
    "--api_key_path",
    help="The path to the OpenAI API key",
    type=str,
    default="./api_key",
)

# Parse command line arguments
args = parser.parse_args()
MODEL = args.model
SOURCE_LANGUAGE = args.source_language
TARGET_LANGUAGE = args.target_language
MAX_TOKENS = args.max_tokens
API_KEY_PATH = args.api_key_path

# Set OpenAI API key
openai.api_key_path = API_KEY_PATH

# Global variables and prompt
prompt = f"""
You are being called as part of a translation pipeline. Here is a description of
your task:
1. You are translating {SOURCE_LANGUAGE} to {TARGET_LANGUAGE}.
2. Your inputs are lists of sentences in {SOURCE_LANGUAGE}. They are formatted
as JSON/Python lists: for instance ['foo', 'bar'].
3. Your outputs should be lists of sentences in {TARGET_LANGUAGE}, formatted
analagously to the inputs: for instance ['baz', 'qux'].
4. Your outputs will be read in using Python's eval() method, so please escape
any special characters you use, consistent with Python.
5. Please ensure that you output a list with the same number of elements as the
input, and that each element of the output is a translation of the corresponding
element of the input.
6. You may break up or combine sentences as you see fit, but please try to
match the inputs: for instance, if you split a sentence into two sentences
during translation, these sentences should be returned as a single element of
the output array, and conversely if you combine two sentences, please still
break the output into two elements.
7. If you encounter the token <NEWLINE>, please return it verbatim.
8. Please do not change the order of the sentences, as the translations will be
interleaved with the original sentences in post-processing.
9. Please ensure that names and other transliterated words are spelled
consistently across sentences.
"""

# Parse command line arguments
filepath = sys.argv[1]
outfile = filepath.split(".")[0] + "_translated.txt"

# Read file into lines
with open(filepath, "r") as f:
    if filepath.endswith(".txt"):
        full_text = f.read()
    elif filepath.endswith(".docx") or filepath.endswith(".doc"):
        doc = docx.Document(filepath)
        full_text = "\n".join([p.text for p in doc.paragraphs])
    else:
        raise ValueError(f"Filetype {filepath.split('.')[-1]} not supported")

# Parse file into sentences:
full_text = full_text.replace("\n", "\n<NEWLINE>\n")
full_text = full_text.replace("\t", "TAB")
sentences = re.split(r"[.!\n]", full_text)
sentences = [s.strip() for s in sentences]
sentences = [s for s in sentences if len(s) > 0]

# Turn sentences array into chunks so we don't exceed the max token limit
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
prompt_tokens = len(tokenizer(prompt)["input_ids"])
chunks = []
chunk = []
chunk_length = prompt_tokens
total_tokens = prompt_tokens
for sentence in sentences:
    n_tokens = len(tokenizer(sentence)["input_ids"])
    if chunk_length + n_tokens > MAX_TOKENS:
        chunks.append(chunk)
        chunk = []
        total_tokens += chunk_length
        chunk_length = prompt_tokens
    chunk.append(sentence)
    chunk_length += n_tokens
chunks.append(chunk)  # Add the last chunk
total_tokens += chunk_length
print(
    f"Broke input into {len(chunks)} chunks spanning {len(sentences)} sentences and {total_tokens} tokens.",
    file=sys.stderr,
)

# API call:
responses = []
for chunk in chunks:
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": str(chunk)},
            # str(chunk) is JSON-formatted by default
        ],
        temperature=0.0,
    )
    response_message = response["choices"][0]["message"]["content"]
    responses.append(response_message)

    # Check for weirdness
    if response["choices"][0]["finish_reason"] == "max_tokens":
        print(
            f"WARNING: Reached max tokens for chunk {chunk}. Translation may be incomplete.",
            file=sys.stderr,
        )

# Print interleaved responses
with open(outfile, "w") as f:
    for i, (chunk, response) in enumerate(zip(chunks, responses)):
        try:
            translations = eval(response)  # Get a list
            if len(chunk) != len(translations):
                print(
                    f"WARNING: Chunk {i} has {len(chunk)} sentences but translation has {len(translations)} sentences.",
                    file=sys.stderr,
                )
            for og, tr in zip(chunk, translations):
                if og == tr == "<NEWLINE>":
                    print("---", file=f)
                else:
                    print(f"{og}\n{tr}\n", file=f)
        except:
            print("##### FALLBACK #####", file=f)
            print(f"{chunk}\n{response}", file=f)
            print("##### END FALLBACK #####", file=f)
