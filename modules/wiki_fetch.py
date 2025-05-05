import wikipedia
import os
import logging

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WikipediaFetcher:
    def __init__(self, language: str = "en"):
        wikipedia.set_lang(language)

    def search_articles(self, query: str, k: int = 5) -> list:
        try:
            results = wikipedia.search(query, results=k)
            return results
        except Exception as e:
            logging.error(f"Error searching for '{query}': {e}")
            return []

    def fetch_article(self, title: str) -> str:
        try:
            page = wikipedia.page(title)
            self.save_locally([page.content], [title])
            return page.content
        except wikipedia.DisambiguationError as e:
            logging.error(
                f"Disambiguation error for '{title}'. Options: {e.options}")
        except wikipedia.PageError:
            logging.error(f"Page '{title}' not found.")
        except Exception as e:
            logging.error(f"Unexpected error while fetching '{title}': {e}")
        return None

    def fetch_multiple_articles(self, titles: list) -> list:
        articles = []
        for title in titles:
            logging.info(f"Fetching '{title}'...")
            content = self.fetch_article(title)
            if content:
                articles.append(content)
            else:
                logging.error(f"Failed to fetch '{title}'.")
        self.save_locally(articles, titles)
        return articles

    def save_locally(self, articles: list, titles: list):
        directory = os.environ.get("DOCS_PATH", "./wiki_docs")
        if not os.path.exists(directory):
            os.makedirs(directory)
        for i, article in enumerate(articles):
            with open(os.path.join(directory, f"{titles[i]}.txt"), "w", encoding="utf-8") as f:
                f.write(article)
