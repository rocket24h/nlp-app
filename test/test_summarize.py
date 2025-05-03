import os
import sys
from dotenv import load_dotenv

# Adds the parent directory (Finals/) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.kg_index import load_KG_from_config
load_dotenv()

def load_docs(path: str) -> list[(str, str)]:
    text_data = []
    for doc in os.listdir(path):
        if doc.endswith('.txt'):
            # print("Reading", doc)
            with open(os.path.join(path, doc), 'r', encoding='utf-8') as file:
                a = file.read()
                # print(a)
                text_data.append((doc,a))

    return text_data

def save_as_text(content: str, path: str):
    '''
        Save content to as file text
    '''
    with open(path, 'w') as file:
        file.write(content)

if __name__ == "__main__":
    docs_path = os.environ.get("DOCS_PATH", "wiki_docs/")

    summarized_docs_path = docs_path[:-1] + '_summarized'
    if not os.path.exists(summarized_docs_path):
        os.makedirs(summarized_docs_path)

    data = load_docs(docs_path)

    kg = load_KG_from_config()
    print("KG initialized")

    for doc, content in data:
        print("Summarizing ", doc)
        summarized_content = kg.summarize(content)  # Summarize document
        save_path = os.path.join(summarized_docs_path, doc)
        save_as_text(summarized_content, save_path)