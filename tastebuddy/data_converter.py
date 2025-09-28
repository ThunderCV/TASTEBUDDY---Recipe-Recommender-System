import pandas as pd
from langchain_core.documents import Document

class DataConverter:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def convert(self):
        # Load the CSV and select relevant columns: name, rating, description, and review
        df = pd.read_csv(self.file_path)[["name", "review"]]

        # Convert each row into a Document object
        docs = [
            Document(
                page_content=row['review'],  # Main content is the review text
                metadata={
                    "recipe_name": row["name"],  # Recipe title as metadata
                }
            )
            for _, row in df.iterrows()
        ]

        return docs