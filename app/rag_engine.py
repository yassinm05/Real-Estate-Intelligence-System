import os
import torch
import chromadb
from chromadb import PersistentClient
from chromadb.api.types import Documents, Embeddings, EmbeddingFunction
from transformers import AutoTokenizer, AutoModel
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv(override=True)

class DistilBERTMeanPoolingEmbeddingFunction(EmbeddingFunction):
    """Custom embedding class required by ChromaDB to translate text."""
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, input: Documents) -> Embeddings:
        encoded_input = self.tokenizer(
            input, padding=True, truncation=True, return_tensors='pt',
            max_length=512, return_token_type_ids=False 
        ).to(self.device)

        with torch.no_grad():
            encoded_input.pop("token_type_ids", None)
            model_output = self.model(**encoded_input)

        token_embeddings = model_output[0] 
        attention_mask = encoded_input['attention_mask']
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask

        return mean_pooled.cpu().numpy().tolist()


class RealEstateRAG:
    def __init__(self):
        print("Initializing RAG Engine...")
        
        
        if not os.environ.get("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY missing from environment variables.")

        
        self.chroma_path = r"C:\Users\w\Desktop\real_estate_rag_ecosystem\chroma_db" 
        self.model_dir = r"C:\Users\w\Desktop\real_estate_rag_ecosystem\models\my_real_estate_distilbert"
        self.collection_name = "seattle_airbnb_inventory"

        
        self.embedder = DistilBERTMeanPoolingEmbeddingFunction(self.model_dir)

        
        self.chroma_client = PersistentClient(path=self.chroma_path)
        self.collection = self.chroma_client.get_collection(
            name=self.collection_name,
            embedding_function=self.embedder
        )

        
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, max_retries=3)

        # 4. Define Prompt Template
        template = """
        You are an expert, highly professional Real Estate and Airbnb Agent in Seattle. 
        A client has asked you a question. 
        
        You must answer their question based ONLY on the following property database context. 
        Do not invent properties, prices, or amenities that are not in the context.
        If the context does not contain relevant properties to answer the question, politely inform the client.
        
        When recommending a property, always mention the Listing ID, the price, and why it fits their request based on the review.

        PROPERTY DATABASE CONTEXT:
        {context}

        CLIENT QUESTION: 
        {question}
        
        AGENT RESPONSE:
        """
        self.prompt = ChatPromptTemplate.from_template(template)
        self.rag_chain = self.prompt | self.llm | StrOutputParser()
        

    def get_recommendation(self, user_query: str, n_results: int = 5) -> dict:
        """Handles retrieval, formatting, and generation."""
        # Retrieve
        results = self.collection.query(query_texts=[user_query], n_results=n_results)
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        
        # Format Context
        context_string = ""
        for i in range(len(documents)):
            meta = metadatas[i]
            context_string += f"\n--- Property Option {i+1} ---\n"
            context_string += f"Listing ID: {meta.get('listing_id', 'Unknown')}\n"
            context_string += f"Price: ${meta.get('price', 'Unknown')}/night\n"
            context_string += f"Bedrooms: {meta.get('bedrooms', 'Unknown')}\n"
            context_string += f"Neighborhood: {meta.get('neighbourhood_cleansed', 'Unknown')}\n"
            context_string += f"Review Snippet: {documents[i]}\n"

        # Generate
        response = self.rag_chain.invoke({
            "context": context_string,
            "question": user_query
        })
        
        return {
            "agent_response": response,
            "raw_context": context_string
        }