import wikipedia
import os

from dotenv import load_dotenv
load_dotenv()


class WikipediaFetcher:
    def __init__(self, language: str = "en"):
        """
        Initialize the Wikipedia fetcher.

        Args:
            language (str): Wikipedia language edition. Default is "en" (English).
        """
        wikipedia.set_lang(language)

    def fetch_article(self, title: str) -> str:
        """
        Fetch the content of a Wikipedia article by its title.

        Args:
            title (str): The title of the Wikipedia article.

        Returns:
            str: The content of the article, or None if fetch failed.
        """
        try:
            page = wikipedia.page(title)
            return page.content
        except wikipedia.DisambiguationError as e:
            print(f"Disambiguation error for '{title}'. Options: {e.options}")
        except wikipedia.PageError:
            print(f"Page '{title}' not found.")
        except Exception as e:
            print(f"Unexpected error while fetching '{title}': {e}")
        return None

    def fetch_multiple_articles(self, titles: list) -> list:
        """
        Fetch multiple Wikipedia articles.

        Args:
            titles (list): A list of Wikipedia article titles.

        Returns:
            list: List of successfully fetched article texts.
        """
        articles = []
        for title in titles:
            print(f"Fetching '{title}'...")
            content = self.fetch_article(title)
            if content:
                articles.append(content)
            else:
                print(f"Failed to fetch '{title}'.")
        return articles

    def save_locally(self, articles: list, titles: list):
        """
        Save fetched articles to local files.

        Args:
            articles (list): List of article texts.
            titles (list): List of article titles.
        """

        directory = os.environ.get("DOCS_PATH", "./wiki_docs")
        if not os.path.exists(directory):
            os.makedirs(directory)
        for i, article in enumerate(articles):
            with open(os.path.join(directory, f"{titles[i]}.txt"), "w", encoding="utf-8") as f:
                f.write(article)
