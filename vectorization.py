from pathlib import Path

import anyio

import app
from core.models.doc_minillm import DocMinillm
from core.vec_db.pgvector.data import Process as PdfProcess
from core.vec_db.pgvector.main import Operator as PgvecDB


class VectorizationService:
    """
    Service for vectoring documents and images.

    This class provides methods to process and vectorized PDF documents and images,
    then save the vectorized data into vector databases.

    Methods:
        run(data_folder: str, types: Literal["pdf", "img"] = "pdf") -> None:
            Process and vectorize the data in the specified folder based on the type.
    """

    def __init__(self, text_emb_model: DocMinillm, recreate: bool = False) -> None:
        """
        Initialize the VectorizationService with text and image embedding models.

        Args:
            text_emb_model (MinillmModel): The text embedding model.
            recreate (bool) : Recreate the vector database.
        """
        self.text_emb_service = text_emb_model
        self.pgvec_db = None

        self.pdf_converter = PdfProcess()
        self.create_pgvecdb(recreate=recreate)

    def create_pgvecdb(self, recreate: bool):
        self.pgvec_db = PgvecDB(recreate_table=recreate)

    def run(self, data_folder: str, recreate: bool = False) -> None:
        """
        Process and vectorized the data in the specified folder based on the type.

        Args:
            data_folder (str): Path to the folder containing data.
            operate (Literal["pdf", "img"], optional): Type of data to process.

        Raises:
            TypeError: If the operate type is not supported.
        """
        save_dir = Path(data_folder)
        try:
            files = [f for f in save_dir.glob("**/*") if f.is_file()]
            file_count = 0
            data_count = 0
            no_content_file = list()
            final_doc_vectors = list()

            if not app.tasks_status.get(save_dir.name):
                app.tasks_status.update({save_dir.name: dict()})
            app.tasks_status[save_dir.name] = {
                "filename": None,
                "is_converted": False,
                "is_empty": False,
                "is_embed": False,
                "write_pgvector": False,
                "task": False,
            }

            for file in files:
                app.tasks_status[save_dir.name].update({"filename": str(file)})
                document, num_data = self.pdf_converter.run(file=file)
                if not num_data:
                    no_content_file.append(str(file))
                    app.tasks_status[save_dir.name].update({"is_empty": True})
                    continue
                app.tasks_status[save_dir.name].update({"is_converted": True})
                document_vectors = self.text_emb_service.run(data=document)
                app.tasks_status[save_dir.name].update({"is_embed": True})
                final_doc_vectors.extend(document_vectors)
                file_count += 1
                data_count += num_data

            print(
                f"Total file: {len(files)}, Convert file: {file_count}, num_data: {data_count}"
            )
            print(f"No content file: {no_content_file}\nnum: {len(no_content_file)}")
            print(f"Final doc vectors: {len(final_doc_vectors)}")

            try:
                if recreate:
                    self.create_pgvecdb(recreate=recreate)
                self.pgvec_db.save(document_vectors=final_doc_vectors)
                app.tasks_status[save_dir.name].update({"write_pgvector": True})
            except BaseException:
                raise RuntimeError from "Can not write data into Vector DB"

            app.tasks_status[save_dir.name].update({"task": True})
            app.tasks_status[save_dir.name].update({"info": dict()})
            app.tasks_status[save_dir.name]["info"] = {
                "Total_file": len(files),
                "Convert_file": file_count,
                "Num_data": data_count,
                "No_content_file": no_content_file,
            }

        except BaseException as e:
            raise RuntimeError from e

    async def async_run(self, data_folder: str, recreate: bool):
        await anyio.to_thread.run_sync(self.run, data_folder, recreate)


if __name__ == "__main__":
    doc_model = DocMinillm()
    Vec_Service = VectorizationService(text_emb_model=doc_model)

    Vec_Service.run(data_folder="inno_pdf/EP")
