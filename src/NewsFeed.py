import json
import random


class NewsFeed:
    def __init__(self, news_dataset_path: str):
        print(f"News feed path is {news_dataset_path}")
        self.path = news_dataset_path
        self.news_items = self.read_jsonl_file(self.path)

    def read_jsonl_file(self, filepath):
        """
        Reads a JSONL file and returns a list of Python dictionaries,
        where each dictionary represents a JSON object from a line in the file.
        """
        data = []
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    # Strip whitespace and check if the line is not empty
                    stripped_line = line.strip()
                    if stripped_line:
                        try:
                            json_object = json.loads(stripped_line)
                            data.append(json_object)
                        except json.JSONDecodeError as e:
                            print(
                                f"Error decoding JSON on line: {stripped_line}. Error: {e}"
                            )
        except FileNotFoundError:
            print(f"Error: File not found at {filepath}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        return data

    def get_random_news(self, nr_of_items: int) -> list[dict]:
        """
        Get a list of news items from the dataset.
        """
        return random.sample(self.news_items, nr_of_items)

    def get_random_news_str(self, nr_of_items: int) -> str:
        """
        Get a string representation of a list of news items.
        """

        news_items = self.get_random_news(nr_of_items)
        msg = ""

        for i, news_item in enumerate(news_items, start=1):
            msg += f"""
                ID: {i}
                Title: {news_item["headline"]}
                Category: {news_item["category"]}
                Description: {news_item["short_description"]}\n\n
            """
        return msg
