import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()

# Free Open-source model load ho raha hai (No API Key Required!)
try:
    model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    print(f"Model load karne me dikkat aayi: {e}")

class SearchRequest(BaseModel):
    query_id: str
    query: str
    candidates: list[str]

class SearchResponse(BaseModel):
    ranking: list[int]

@app.post("/rank", response_model=SearchResponse)
async def rank_candidates(payload: SearchRequest):
    if not payload.candidates:
        return {"ranking": []}
    
    try:
        # 1. Query aur candidates dono ko list me daala
        all_texts = [payload.query] + payload.candidates
        
        # 2. Free local model se embeddings generate kiye
        all_embeddings = model.encode(all_texts)
        
        # 3. Vectors alag kiye
        query_vector = np.array(all_embeddings[0])
        candidate_vectors = np.array(all_embeddings[1:])
        
        # 4. Cosine Similarity calculate ki
        dot_products = np.dot(candidate_vectors, query_vector)
        candidate_norms = np.linalg.norm(candidate_vectors, axis=1)
        query_norm = np.linalg.norm(query_vector)
        
        similarities = dot_products / (candidate_norms * query_norm + 1e-9)
        
        # 5. Top 3 highest indices nikale
        top_3_indices = np.argsort(similarities)[-3:][::-1]
        
        return {"ranking": top_3_indices.tolist()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Error: {str(e)}")
