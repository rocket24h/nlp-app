from dotenv import set_key

# Define default values for environment variables
default_env_values = {
    "GOOGLE_API_KEY": None,  # Mandatory field
    "DOCS_PATH": "wiki_docs/",
    "PERSIST_PATH": "graph_store/",
    "EMBEDDING_MODEL": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "LLM_MODEL": "gemini-2.0-flash",
    "LLM_TEMPERATURE": "1.0",
    "LLM_MAX_TOKENS": "2000",
    "SEARCH_DEPTH": "2",
    "CHAT_HISTORY_PATH": "chat_history.json",
    "NEO4J_URL": None,
    "NEO4J_USERNAME": "neo4j",
    "NEO4J_PASSWORD": None,
    "NEO4J_DATABASE": "neo4j",
}

# Function to prompt user input with default values


def prompt_user_input(key, default_value, is_mandatory=False):
    while True:
        user_input = input(f"{key} - [{default_value}]: ").strip()
        if user_input:
            return user_input
        elif not is_mandatory:
            return default_value
        else:
            print(f"{key} is mandatory. Please provide a value.")


# Prompt user for each environment variable
print("No .env file detected. Please provide the following settings.")
env_values = {}
for key, default_value in default_env_values.items():
    is_mandatory = key in ["GOOGLE_API_KEY", "NEO4J_URL", "NEO4J_PASSWORD"]
    if default_value is None:
        # If no default value, prompt for mandatory input
        env_values[key] = prompt_user_input(key, "", is_mandatory)
    else:
        # If default value exists, prompt with default
        env_values[key] = prompt_user_input(key, default_value, is_mandatory)

# Save the provided values to a new .env file
print("\nCreating .env file with the provided settings...")
with open(".env", "w") as env_file:
    for key, value in env_values.items():
        env_file.write(f"{key}={value}\n")

print("Settings saved successfully to .env!")
