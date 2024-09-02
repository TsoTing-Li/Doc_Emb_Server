import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass

import uvicorn
from fastapi import (
    BackgroundTasks,
    FastAPI,
    Response,
    WebSocket,
    WebSocketDisconnect,
    status,
)

import schema
import validator
from core.models.doc_minillm import DocMinillm
from vectorization import VectorizationService


@dataclass
class Params:
    name: str | None = None
    model: DocMinillm | None = None
    is_loaded: bool = None


ModelServer = Params()


@asynccontextmanager
async def lifespan(app: FastAPI):
    ModelServer.model = DocMinillm()
    ModelServer.name = "doc_minillm"
    ModelServer.is_loaded = True

    global vec_service
    vec_service = VectorizationService(text_emb_model=ModelServer.model, recreate=True)

    yield

    ModelServer.model = None
    ModelServer.name = None
    ModelServer.is_loaded = False


app = FastAPI(lifespan=lifespan)

tasks_status = dict()


@app.post("/embed/doc/", tags=["Embed"])
async def post_embed_doc(
    background_task: BackgroundTasks, request_data: schema.PostEmbedDoc
):
    validator.PostEmbedDoc(data_folder=request_data.data_folder)

    background_task.add_task(
        vec_service.async_run,
        f"upload_pdf/{request_data.data_folder}",
        request_data.recreate,
    )

    return Response(
        content=json.dumps({"task_id": request_data.data_folder}),
        status_code=status.HTTP_201_CREATED,
        media_type="application/json",
    )


@app.websocket("/ws/{task_id}")
async def websocket(websocket: WebSocket, task_id: str):
    await websocket.accept()
    previous_filename = None

    try:
        while True:
            progress = tasks_status.get(task_id)
            if progress is None:
                break
            if progress["filename"] != previous_filename:
                try:
                    await websocket.send_json(progress)
                    previous_filename = progress["filename"]
                except BaseException:
                    break

            if progress["task"] is True:
                await websocket.send_json(progress["info"])
                break

            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        print(f"WebSocket connection closed for task: {task_id}")
    finally:
        await websocket.close()


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8777, reload=True)
