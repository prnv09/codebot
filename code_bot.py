import pandas as pd
from pathlib import Path
# from utils.embeddings_utils import get_embedding
from transformers import AutoTokenizer, AutoModel
import torch
import streamlit as st
DEF_PREFIXES = ['def ', 'async def ']
NEWLINE = '\n'
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


def get_function_name(code):
    """
    Extract function name from a line beginning with 'def' or 'async def'.
    """
    for prefix in DEF_PREFIXES:
        if code.startswith(prefix):
            return code[len(prefix): code.index('(')]


def get_until_no_space(all_lines, i):
    """
    Get all lines until a line outside the function definition is found.
    """
    ret = [all_lines[i]]
    for j in range(i + 1, len(all_lines)):
        if len(all_lines[j]) == 0 or all_lines[j][0] in [' ', '\t', ')']:
            ret.append(all_lines[j])
        else:
            break
    return NEWLINE.join(ret)


def get_functions(filepath):
    """
    Get all functions in a Python file.
    """
    with open(filepath, 'r') as file:
        all_lines = file.read().replace('\r', NEWLINE).split(NEWLINE)
        for i, l in enumerate(all_lines):
            for prefix in DEF_PREFIXES:
                if l.startswith(prefix):
                    code = get_until_no_space(all_lines, i)
                    function_name = get_function_name(code)
                    yield {
                        'code': code,
                        'function_name': function_name,
                        'filepath': filepath,
                    }
                    break


def extract_functions_from_repo(code_root):
    """
    Extract all .py functions from the repository.
    """
    code_files = list(code_root.glob('**/*.py'))

    num_files = len(code_files)
    #print(f'Total number of .py files: {num_files}')

    if num_files == 0:
        print('Verify openai-python repo exists and code_root is set correctly.')
        return None

    all_funcs = [
        func
        for code_file in code_files
        for func in get_functions(str(code_file))
    ]

    num_funcs = len(all_funcs)
    #print(f'Total number of functions extracted: {num_funcs}')

    return all_funcs

# Set user root directory to the 'openai-python' repository
#root_dir = Path.home()
root_dir = Path.cwd()

# Assumes the 'openai-python' repository exists in the user's root directory
code_root = root_dir / 'code'
# Assumes the 'openai-python' repository exists in the user's root directory
# Extract all functions from the repository
all_funcs = extract_functions_from_repo(code_root)


# def get_embedding(text, model="text-embedding-ada-002"):
#    text = text.replace("\n", " ")
#    return client.embeddings.create(input = [text], 
# model=model).data[0].embedding

def get_embedding(text, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    text = text.replace("\n", " ")
    
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    # Get the embeddings from the model
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the embeddings from the last hidden state
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    
    return embeddings


#print(all_funcs)
df = pd.DataFrame(all_funcs)
#print(df.head)
df['code_embedding'] = df['code'].apply(lambda x: get_embedding(x))
df['filepath'] = df['filepath'].map(lambda x: Path(x).relative_to(code_root))
#df.to_csv("./code_search_openai-python.csv", index=False)
df.head()

# from utils.embeddings_utils import cosine_similarity
import numpy as np
def custom_cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    return dot_product / (norm_vec1 * norm_vec2)

def search_functions(df, code_query, n=3, pprint=True, n_lines=7):
    embedding = get_embedding(code_query)
    df['similarities'] = df.code_embedding.apply(lambda x: custom_cosine_similarity(x, embedding))

    res = df.sort_values('similarities', ascending=False).head(n)

    # if pprint:
    #     for r in res.iterrows():
    #         print(f"{r[1].filepath}:{r[1].function_name}  score={round(r[1].similarities, 3)}")
    #         print("\n".join(r[1].code.split("\n")[:n_lines]))
    #         print('-' * 70)

    return res

query = st.text_input(label="Ask to codeBot")
if query:

    res = search_functions(df, query, n=3)
    st.code(res['code'].iloc[0])