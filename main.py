import os
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()

# Initialize OpenAI client (assumes OPENAI_API_KEY is in your environment variables)
client = OpenAI()

class SearchRequest(BaseModel):
    query_id: str
    query: str
    candidates: list[str]

class SearchResponse(BaseModel):
    ranking: list[int]

def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Batches text strings to OpenAI's text-embedding-3-small model."""
    try:
        response = client.embeddings.create(
            input=texts,
            model="text-embedding-3-small"
        )
        return [data.embedding for data in response.data]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API Error: {str(e)}")

@app.post("/rank", response_model=SearchResponse)
async def rank_candidates(payload: SearchRequest):
    if not payload.candidates:
        return {"ranking": []}
        
    # 1. Batch embed the query and all candidates together to save API roundtrips
    all_texts = [payload.query] + payload.candidates
    all_embeddings = get_embeddings(all_texts)
    
    # 2. Separate query vector from candidate vectors
    query_vector = np.array(all_embeddings[0])
    candidate_vectors = np.array(all_embeddings[1:])
    
    # 3. Compute Cosine Similarity using numpy
    # Cosine Similarity = (A · B) / (||A|| * ||B||)
    dot_products = np.dot(candidate_vectors, query_vector)
    candidate_norms = np.linalg.norm(candidate_vectors, axis=1)
    query_norm = np.linalg.norm(query_vector)
    
    # Prevent division by zero if an embedding is somehow empty
    sims = dot_products / (candidate_norms * query_norm + 1e-9)
    
    # 4. Extract top 3 highest scoring indices
    # np.argsort returns ascending order; [-3:] gets the top 3; [::-1] makes it descending
    top_k_indices = np.argsort(sims)[-3:][::-1]
    
    return {"ranking": top_k_indices.tolist()}
