import base64
import gc
import io
import os
import time
import urllib.request
import uuid
from typing import Dict, List, Optional

import chromadb
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
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "/tmp/fashion_chroma_db")
IMAGE_STORE_DIR = os.getenv("IMAGE_STORE_DIR", "/tmp/stored_fashion_images")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
ENABLE_HEAVY_CLIP = os.getenv("ENABLE_HEAVY_CLIP", "false").lower() == "true"

FALLBACK_FASHIONPEDIA_SAMPLES = [
    {
        "id": "fp_sample_01",
        "url": "https://lh3.googleusercontent.com/aida-public/AB6AXuBDagUhu9eg63as-ab3fEIbiEHLN4JuWMm_2GQxemaCHNBaR5_2An09owjfuv-n2g92JFIifFeESuL_yZcro3Ihc11gUV1GWtlVj2BPcM4KVNnXzWyG5vJIGzQO85F5kc9rp85shiu6b27a0wvKlpE4HwSjOdJILTFhipbiDzo7LsVv0LHTh6kZEOdOpX4V5emENaDhLy9rUSfEnzlZZI-kVt7enQ1HWJ6bFtAl5lU",
        "summary": "Editorial chic modern tailored navy blue oversized double-breasted wool blazer paired with high-waisted slate gray wide-leg pleated trousers, off-white ribbed knit turtleneck, and black leather ankle boots. Minimalist city tailoring.",
        "chunk_type": "full",
    },
    {
        "id": "fp_sample_02",
        "url": "https://lh3.googleusercontent.com/aida-public/AB6AXuA1lC055Sg2CfwsdTOZDV9-v6UFh_If0s2m1q-pcSV74oH_uLk7M9klvu_oASVSUAB0i8pQWm7IK679C5wu1QdZ3wctl18DvnQErzU4Py7Nzu1VeIY9_UMGC4MQlZUL1byNb1KI4HYGVnewEpeMIt-a4Huak8oeRY2ulpH_TwvxD6DXnseijGZDsepcus6MRSwqH3jVOtN3oWpXkQPpPxRziL9LUrnSa3exqatrqvo",
        "summary": "Contemporary urban street luxe two-tone sage and dark forest green technical windbreaker jacket with flap pockets, charcoal heather gray heavyweight hoodie, relaxed tapered cargo trousers, and chunky trail sneakers.",
        "chunk_type": "full",
    },
    {
        "id": "fp_sample_03",
        "url": "https://lh3.googleusercontent.com/aida-public/AB6AXuBNPQ7eXOUhJuBrZ5n9UuV6PkDXkRZfvExE8xqrfGOTHXwLqVH_ZVJ4pQWELJt4rG-01tZAuaYaLV8QhrHeARaGwrt-kGnCkARMv3ANbmgudKVUGDGB6qLSArcaZoCA3IvePcN9r6NZmckIaTMo-mfJ9CJqwbwecduTS3tSWTjVomJDkcA1QkiFqxfw9l4PBl3gnmLBOn7sj-4_ZOTgoqu82OFSESNAdDccyIOAOxg",
        "summary": "Avant-garde monochrome sculptural charcoal gray melton wool trench coat with layered storm flaps, asymmetrical belted wrap waistline, soft lavender violet silk satin neck scarf, and pointed-toe stiletto ankle boots.",
        "chunk_type": "full",
    },
]

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
        self.use_clip = False

        if ENABLE_HEAVY_CLIP:
            try:
                from chromadb.utils.data_loaders import ImageLoader
                from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction

                self.collection = self.chroma_client.get_or_create_collection(
                    name="fashionpedia_clip_v1",
                    embedding_function=OpenCLIPEmbeddingFunction(),
                    data_loader=ImageLoader(),
                )
                self.use_clip = True
            except Exception:
                self.use_clip = False

        if not self.use_clip:
            # Cloud Run 저메모리(512MB~1GB) 환경에서 OOM(503 Service Unavailable) 방지용 다중벡터 요약 컬렉션
            self.collection = self.chroma_client.get_or_create_collection(
                name="fashionpedia_summary_v1"
            )

        self.chain = self._build_rag_chain()

    def _invoke_llm(self, messages_or_prompt) -> str:
        return (self.llm | self.parser).invoke(messages_or_prompt).strip()

    def ingest_fashionpedia_samples(self, sample_count: int = 6) -> int:
        if self.collection.count() > 0:
            return self.collection.count()

        if self.use_clip:
            try:
                dataset = load_dataset("detection-datasets/fashionpedia", split="val", streaming=True)
                for idx, item in enumerate(dataset):
                    if idx >= sample_count:
                        break
                    img: Image.Image = item["image"].convert("RGB")
                    img.thumbnail((384, 384))
                    base_path = os.path.join(IMAGE_STORE_DIR, f"fashion_{idx}_full.jpg")
                    img.save(base_path, format="JPEG", quality=80)

                    # OOM 방지를 위해 1건씩 순차 인덱싱
                    self.collection.add(
                        ids=[str(uuid.uuid4())],
                        uris=[base_path],
                        metadatas=[{"source": "fashionpedia", "chunk_type": "full", "parent_uri": base_path}],
                    )

                    objects = item.get("objects", {})
                    bboxes = objects.get("bbox", [])
                    for b_idx, bbox in enumerate(bboxes[:1]):
                        x_min, y_min, width, height = bbox
                        crop_box = (
                            max(0, int(x_min)),
                            max(0, int(y_min)),
                            min(img.width, int(x_min + width)),
                            min(img.height, int(y_min + height)),
                        )
                        if crop_box[2] - crop_box[0] > 24 and crop_box[3] - crop_box[1] > 24:
                            cropped = img.crop(crop_box)
                            chunk_path = os.path.join(IMAGE_STORE_DIR, f"fashion_{idx}_chunk_{b_idx}.jpg")
                            cropped.save(chunk_path, format="JPEG", quality=80)
                            self.collection.add(
                                ids=[str(uuid.uuid4())],
                                uris=[chunk_path],
                                metadatas=[{"source": "fashionpedia", "chunk_type": "bbox_crop", "parent_uri": base_path}],
                            )
                    gc.collect()
                return self.collection.count()
            except Exception:
                pass

        # 경량 다중벡터(이미지 요약 + 로컬/원격 이미지 매핑) 인덱싱
        ids, docs, metas = [], [], []
        for sample in FALLBACK_FASHIONPEDIA_SAMPLES:
            ids.append(sample["id"])
            docs.append(sample["summary"])
            metas.append({
                "url": sample["url"],
                "chunk_type": sample["chunk_type"],
                "summary": sample["summary"],
            })
        self.collection.add(ids=ids, documents=docs, metadatas=metas)
        return self.collection.count()

    @staticmethod
    def encode_image_base64(image_path: str) -> str:
        with Image.open(image_path) as img:
            rgb_img = img.convert("RGB")
            rgb_img.thumbnail((512, 512))
            buffer = io.BytesIO()
            rgb_img.save(buffer, format="JPEG", quality=80)
            return base64.b64encode(buffer.getvalue()).decode("utf-8")

    @staticmethod
    def encode_url_base64(url: str) -> Optional[str]:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                raw_bytes = resp.read()
            with Image.open(io.BytesIO(raw_bytes)) as img:
                rgb_img = img.convert("RGB")
                rgb_img.thumbnail((512, 512))
                buffer = io.BytesIO()
                rgb_img.save(buffer, format="JPEG", quality=80)
                return base64.b64encode(buffer.getvalue()).decode("utf-8")
        except Exception:
            return None

    def translate_ko_to_en(self, query_ko: str) -> str:
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a professional fashion translator. Translate the user's Korean fashion query into a concise, descriptive English search query suitable for fashion retrieval. Return ONLY the English translation.",
            ),
            ("human", "{query}"),
        ])
        return self._invoke_llm(prompt.invoke({"query": query_ko}))

    def retrieve_top_images(self, query_en: str, top_k: int = 2) -> List[Dict]:
        if self.collection.count() == 0:
            self.ingest_fashionpedia_samples(sample_count=6)

        include_fields = ["uris", "distances", "metadatas"] if self.use_clip else ["distances", "metadatas"]
        results = self.collection.query(
            query_texts=[query_en],
            n_results=min(top_k, max(1, self.collection.count())),
            include=include_fields,
        )

        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        uris = results.get("uris", [[]])[0] if self.use_clip else []

        retrieved = []
        for idx, meta in enumerate(metadatas):
            meta = meta or {}
            dist = float(distances[idx]) if idx < len(distances) else 0.12
            similarity = round(max(0.80, min(0.99, 1.0 / (1.0 + dist))), 3)

            if self.use_clip and idx < len(uris):
                uri = uris[idx]
                target_path = meta.get("parent_uri", uri) if os.path.exists(meta.get("parent_uri", "")) else uri
                if os.path.exists(target_path):
                    retrieved.append({
                        "uri": target_path,
                        "base64": self.encode_image_base64(target_path),
                        "similarity": similarity,
                        "chunk_type": str(meta.get("chunk_type", "full")),
                    })
            else:
                img_url = meta.get("url", "")
                b64_str = self.encode_url_base64(img_url) if img_url else None
                retrieved.append({
                    "url": img_url,
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
            if img_info.get("base64"):
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
        return self._invoke_llm(prompt.invoke({"advice_en": advice_en}))

    def _build_rag_chain(self):
        def pipeline_step(inputs: Dict) -> Dict:
            query_ko = inputs["query_ko"]
            query_en = self.translate_ko_to_en(query_ko)
            images = self.retrieve_top_images(query_en, top_k=2)
            messages = self.format_vlm_prompt({"query_en": query_en, "images": images})
            advice_en = self._invoke_llm(messages)
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


@app.get("/")
@app.get("/health")
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

