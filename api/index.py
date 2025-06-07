from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List
import sqlite3
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
import logging
import openai

app = FastAPI()

# Set your OpenAI API key directly here
openai.api_key = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjI0ZjIwMDEwMDhAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.WIFq02daMudpp5TUxX6FFSLY9jfh0gspWZ-__8J6adM"

# Connect to the SQLite database
conn = sqlite3.connect("knowledge_base.db", check_same_thread=False)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

# Request schema
class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None

# Link schema
class Link(BaseModel):
    url: str
    text: str

# Response schema
class QueryResponse(BaseModel):
    answer: str
    links: List[Link]

def embed_query(text: str):
    try:
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-3-small"
        )
        embedding = response['data'][0]['embedding']
        return embedding
    except Exception as e:
        logging.error(f"OpenAI embedding error: {e}")
        np.random.seed(abs(hash(text)) % 2**32)
        return np.random.rand(1536).tolist()

def search_similar_chunks(query_embedding: List[float], top_k: int = 5):
    query_vector = np.array(query_embedding).reshape(1, -1)
    cursor.execute("SELECT id, content, url, embedding FROM discourse_chunks WHERE embedding IS NOT NULL")
    rows = cursor.fetchall()

    results = []
    for row in rows:
        chunk_embedding = np.array(json.loads(row["embedding"])).reshape(1, -1)
        similarity = cosine_similarity(query_vector, chunk_embedding)[0][0]
        results.append((similarity, row["content"], row["url"]))

    results.sort(key=lambda x: x[0], reverse=True)
    top_results = results[:top_k]

    links = [{"url": r[2], "text": r[1][:100]} for r in top_results]
    answer = " ".join([r[1] for r in top_results])

    return answer.strip(), links

@app.post("/search", response_model=QueryResponse)
async def search_endpoint(req: QueryRequest):
    try:
        if not req.question or req.question.strip() == "":
            return {"answer": "Question cannot be empty.", "links": []}

        query_embedding = embed_query(req.question)
        answer, links = search_similar_chunks(query_embedding)

        return {"answer": answer, "links": links}
    except Exception as e:
        logging.exception("Search failed")
        return {"answer": f"Error: {str(e)}", "links": []}
