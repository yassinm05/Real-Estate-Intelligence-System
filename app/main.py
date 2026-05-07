from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.rag_engine import RealEstateRAG
import uvicorn


class ChatRequest(BaseModel):
    query: str
    n_results: int = 5 

class ChatResponse(BaseModel):
    agent_response: str
    raw_context: str


app = FastAPI(
    title="Seattle Real Estate RAG API",
    description="An intelligent property recommendation engine powered by DistilBERT, ChromaDB, and Google Gemini.",
    version="1.0.0"
)

rag_service = None

@app.on_event("startup")
async def startup_event():
    """Fires when the server starts to load the heavy ML models into memory."""
    global rag_service
    try:
        rag_service = RealEstateRAG()
    except Exception as e:
        print(f"Failed to initialize RAG Engine: {e}")

@app.post("/api/recommend", response_model=ChatResponse)
async def get_recommendation(request: ChatRequest):
    """
    Accepts a natural language query and returns AI-generated property recommendations.
    """
    if not rag_service:
        raise HTTPException(status_code=503, detail="AI Engine is still initializing or offline.")
    
    try:
        
        result = rag_service.get_recommendation(
            user_query=request.query, 
            n_results=request.n_results
        )
        return ChatResponse(
            agent_response=result["agent_response"],
            raw_context=result["raw_context"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)