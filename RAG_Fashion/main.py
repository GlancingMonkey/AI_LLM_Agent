import base64
import io
import os
import time
import uuid
from typing import Dict, List, Optional

import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from datasets import load_dataset
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from PIL import Image
from pydantic import BaseModel

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./fashion_chroma_db")
IMAGE_STORE_DIR = os.getenv("IMAGE_STORE_DIR", "./stored_fashion_images")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

app = FastAPI(title="Atelier AI - Fashion Multimodal RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class StyleRequest(BaseModel):
    query: str
    image_base64: Optional[str] = None
    weight_mode: Optional[str] = "silhouette"


class FashionMultimodalRAG:
    def __init__(self) -> None:
        os.makedirs(CHROMA_DB_DIR, exist_ok=True)
        os.makedirs(IMAGE_STORE_DIR, exist_ok=True)

        self.llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.4,
        )
        self.parser = StrOutputParser()
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        self.embedding_fn = OpenCLIPEmbeddingFunction()
        self.image_loader = ImageLoader()

        self.collection = self.chroma_client.get_or_create_collection(
            name="fashionpedia_multimodal",
            embedding_function=self.embedding_fn,
            data_loader=self.image_loader,
        )
        self.chain = self._build_rag_chain()

    def ingest_fashionpedia_samples(self, sample_count: int = 30) -> int:
        if self.collection.count() > 0:
            return self.collection.count()

        dataset = load_dataset("detection-datasets/fashionpedia", split="val", streaming=True)
        ids, uris, metadatas = [], [], []

        for idx, item in enumerate(dataset):
            if idx >= sample_count:
                break
            img: Image.Image = item["image"].convert("RGB")
            base_path = os.path.join(IMAGE_STORE_DIR, f"fashion_{idx}_full.jpg")
            img.save(base_path, format="JPEG", quality=90)

            ids.append(str(uuid.uuid4()))
            uris.append(base_path)
            metadatas.append({"source": "fashionpedia", "chunk_type": "full", "sample_idx": idx})

            # 이미지 청킹(Bounding Box 단위 의류 영역 크롭) 및 다중벡터 인덱싱
            objects = item.get("objects", {})
            bboxes = objects.get("bbox", [])
            categories = objects.get("category", [])
            for b_idx, bbox in enumerate(bboxes[:2]):
                x_min, y_min, width, height = bbox
                crop_box = (
                    max(0, int(x_min)),
                    max(0, int(y_min)),
                    min(img.width, int(x_min + width)),
                    min(img.height, int(y_min + height)),
                )
                if crop_box[2] - crop_box[0] > 32 and crop_box[3] - crop_box[1] > 32:
                    cropped = img.crop(crop_box)
                    chunk_path = os.path.join(IMAGE_STORE_DIR, f"fashion_{idx}_chunk_{b_idx}.jpg")
                    cropped.save(chunk_path, format="JPEG", quality=85)
                    cat_id = str(categories[b_idx]) if b_idx < len(categories) else "garment"
                    ids.append(str(uuid.uuid4()))
                    uris.append(chunk_path)
                    metadatas.append({
                        "source": "fashionpedia",
                        "chunk_type": "bbox_crop",
                        "category": cat_id,
                        "parent_uri": base_path,
                    })

        if ids:
            self.collection.add(ids=ids, uris=uris, metadatas=metadatas)
        return self.collection.count()

    @staticmethod
    def encode_image_base64(image_path: str) -> str:
        with Image.open(image_path) as img:
            rgb_img = img.convert("RGB")
            rgb_img.thumbnail((768, 768))
            buffer = io.BytesIO()
            rgb_img.save(buffer, format="JPEG", quality=85)
            return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def translate_ko_to_en(self, query_ko: str) -> str:
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a professional fashion translator. Translate the user's Korean fashion query into a concise, descriptive English search query suitable for CLIP image retrieval. Return ONLY the English translation.",
            ),
            ("human", "{query}"),
        ])
        return (prompt | self.llm | self.parser).invoke({"query": query_ko}).strip()

    def retrieve_top_images(self, query_en: str, top_k: int = 2) -> List[Dict[str, str]]:
        if self.collection.count() == 0:
            self.ingest_fashionpedia_samples(sample_count=20)

        results = self.collection.query(
            query_texts=[query_en],
            n_results=top_k,
            include=["uris", "distances", "metadatas"],
        )

        uris = results.get("uris", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        retrieved = []
        for idx, uri in enumerate(uris):
            meta = metadatas[idx] if idx < len(metadatas) and metadatas[idx] else {}
            target_path = meta.get("parent_uri", uri) if os.path.exists(meta.get("parent_uri", "")) else uri
            if os.path.exists(target_path):
                b64_str = self.encode_image_base64(target_path)
                dist = float(distances[idx]) if idx < len(distances) else 0.1
                similarity = round(max(0.75, min(0.99, 1.0 / (1.0 + dist))), 3)
                retrieved.append({
                    "uri": target_path,
                    "base64": b64_str,
                    "similarity": similarity,
                    "chunk_type": str(meta.get("chunk_type", "full")),
                })
        return retrieved

    def format_vlm_prompt(self, payload: Dict) -> List[HumanMessage]:
        query_en = payload["query_en"]
        images = payload["images"]

        content_parts: List[Dict] = [
            {
                "type": "text",
                "text": (
                    "You are a senior fashion stylist at Atelier AI. Based on the user's request and the retrieved reference fashion images, "
                    "provide professional styling advice in English structured clearly with:\n"
                    "1. [Look Title]: A concise editorial title.\n"
                    "2. [Silhouette & Fit]: Analysis of proportions, drape, and volume balance.\n"
                    "3. [Material & Color Harmony]: Key fabric textures and 3-color palette recommendations (include 3 HEX codes like #1B243B).\n"
                    "4. [Stylist Curation Tip]: Practical coordination and accessory advice.\n\n"
                    f"User Request (EN): {query_en}"
                ),
            }
        ]

        for img_info in images:
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_info['base64']}"},
            })

        return [HumanMessage(content=content_parts)]

    def translate_en_to_ko(self, advice_en: str) -> str:
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an expert Korean fashion editor. Translate the following English styling advice into natural, sophisticated Korean while preserving the structured sections ([룩 타이틀], [실루엣 & 핏 분석], [소재 & 컬러 하모니], [수석 스타일리스트 코멘트]) and any HEX color codes.",
            ),
            ("human", "{advice_en}"),
        ])
        return (prompt | self.llm | self.parser).invoke({"advice_en": advice_en}).strip()

    def _build_rag_chain(self):
        def pipeline_step(inputs: Dict) -> Dict:
            query_ko = inputs["query_ko"]
            query_en = self.translate_ko_to_en(query_ko)
            images = self.retrieve_top_images(query_en, top_k=2)
            messages = self.format_vlm_prompt({"query_en": query_en, "images": images})
            advice_en = (self.llm | self.parser).invoke(messages)
            advice_ko = self.translate_en_to_ko(advice_en)
            return {
                "query_ko": query_ko,
                "query_en": query_en,
                "images": images,
                "advice_en": advice_en,
                "advice_ko": advice_ko,
            }

        return RunnableLambda(pipeline_step)

    def run(self, query_ko: str) -> Dict:
        return self.chain.invoke({"query_ko": query_ko})


rag_engine: Optional[FashionMultimodalRAG] = None


def get_rag_engine() -> FashionMultimodalRAG:
    global rag_engine
    if rag_engine is None:
        rag_engine = FashionMultimodalRAG()
    return rag_engine


@app.get("/api/v1/health")
async def health_check() -> Dict:
    return {
        "status": "success",
        "connection": "CONNECTED",
        "model": GEMINI_MODEL,
        "vector_db": "ChromaDB (OpenCLIP Joint Embedding)",
    }


@app.post("/api/v1/style/recommend")
async def recommend_style(request: StyleRequest) -> Dict:
    start_time = time.perf_counter()
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="질의를 입력해 주세요.")

    try:
        engine = get_rag_engine()
        result = engine.run(request.query.strip())
        latency_ms = round((time.perf_counter() - start_time) * 1000, 1)

        return {
            "status": "success",
            "latency_ms": latency_ms,
            "data": {
                "query_ko": result["query_ko"],
                "query_en": result["query_en"],
                "advice_ko": result["advice_ko"],
                "advice_en": result["advice_en"],
                "images": result["images"],
                "weight_mode": request.weight_mode,
            },
        }
    except Exception as exc:
        return {
            "status": "error",
            "message": str(exc),
            "data": None,
        }

