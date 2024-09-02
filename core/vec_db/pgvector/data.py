import time
from pathlib import Path

from haystack.components.converters import PDFMinerToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter

from tools.logger import config_logger

# init log
LOGGER = config_logger(
    log_name="pgvec_data.log",
    logger_name="pgvec_data",
    default_folder="./log",
    write_mode="w",
    level="debug",
)


class Process:
    """
    Perform data preprocessing for RAG (Retrieval-Augmented Generation).

    This class includes tools for converting PDFs to documents, cleaning the documents,
    and splitting them for further processing.

    Attributes:
        pdf_converter (PyPDFToDocument): Tool for converting PDFs to documents.
        document_cleaner (DocumentCleaner): Tool for cleaning documents.
        document_splitter (DocumentSplitter): Tool for splitting documents.

    Methods:
        run(data_folder: str) -> List[dict]:
            Preprocess data for RAG.
    """

    def __init__(self) -> None:
        """
        Initialize the Process class for data preprocessing.
        """
        LOGGER.info("Init pgvector data process...")
        self._init_tools()
        LOGGER.info("Success init pgvector data process...")

    def _init_tools(self):
        """
        Initialize all tools for data processing.
        """
        # TODO: More flexible
        self.pdf_converter = PDFMinerToDocument()
        LOGGER.info("Success init pdf_converter...")
        self.document_cleaner = DocumentCleaner(
            remove_empty_lines=True,
            remove_extra_whitespaces=True,
            remove_repeated_substrings=False,
        )
        LOGGER.info(f"""Success init document_cleaner , remove_empty_lines :{True}
                     remove_extra_whitespaces:{True}
                     remove_repeated_substrings:{False}""")
        self.document_splitter = DocumentSplitter(
            split_by="word", split_length=150, split_overlap=50
        )
        LOGGER.info("""Success init document_splitter , split_by :"word"
                     split_length:150
                     split_overlap:50""")

    def run(self, file: Path) -> list:
        """
        Preprocess data for RAG.

        Args:
            data_folder (str): The path to the data folder (currently only supports PDF files).

        Returns:
            List[dict]: List of processed documents after cleaning and splitting.
        """
        LOGGER.info(f"Start convert PDF: {str(file)}")
        start = time.time()
        docs = self.pdf_converter.run(
            sources=[file],
            meta={"privacy": 0},
        )
        LOGGER.info(f"Cost: {time.time() - start}")
        clear_docs = self.document_cleaner.run(documents=docs["documents"])
        split_docs = self.document_splitter.run(documents=clear_docs["documents"])
        result = split_docs["documents"]
        LOGGER.info(
            f"Success preprocess data '{str(file)}' for RAG, num: {len(result)} "
        )
        return result, len(result)
