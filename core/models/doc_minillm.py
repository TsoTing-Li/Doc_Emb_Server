from haystack.components.embedders import SentenceTransformersDocumentEmbedder

from core.models.pattern import TextEmbedding
from tools.logger import config_logger

LOGGER = config_logger(
    log_name="doc_embed.log",
    logger_name="doc_embed",
    default_folder="./log",
    write_mode="w",
    level="debug",
)


class DocMinillm(TextEmbedding):
    def __init__(
        self, model_name: str = "hf_models/sentence-transformers/all-MiniLM-L6-v2"
    ) -> None:
        super().__init__(model_name)
        self.model_name = model_name
        self.model = self._load_model()
        self._warm_up()

    def _load_model(self):
        model = SentenceTransformersDocumentEmbedder(model=self.model_name)
        LOGGER.info("Success init DocMinillm")
        return model

    def _warm_up(self):
        self.model.warm_up()
        LOGGER.info("Success warm up DocMinillm")

    def run(self, data: list) -> list:
        emb_docs = self.model.run(documents=data)
        return emb_docs["documents"]
