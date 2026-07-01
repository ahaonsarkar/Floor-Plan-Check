# FastAPI core application with evaluation and chatbot endpoints

import os
import logging
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mock evaluation endpoint
@app.post("/evaluate")
async def evaluate_floorplan(file: UploadFile = File(...)):
    # TODO: Implement evaluation logic
    return {
        "status": "evaluation_not_implemented",
        "message": "Complete implementation in services/image_processor.py and services/evaluation_engine.py"
    }

# Documentation endpoint
@app.get("/docs")
async def docs():
    return "Automated 2D Floor Planning Evaluation API"

# Evaluation endpoint – receives an image, runs the full pipeline, and returns the result
@app.post("/upload-floor-plan")
async def upload_floor_plan(file: UploadFile = File(...)):
    """Upload a floor‑plan image and run the evaluation pipeline.

    This endpoint matches the URL used by the React front‑end (`/upload-floor-plan`).
    It validates the image, reads its bytes, runs the processing pipeline defined in
    ``backend/api/main.py`` and returns the JSON result.
    """
    # Basic validation (same logic as ``validate_image_file`` in api/main.py)
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"Invalid file type: {file.content_type}. Expected image.")
    if file.size > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 50 MiB).")
    
    # Read the uploaded file into memory
    contents = await file.read()
    # Import the processing function lazily to avoid circular import issues
    try:
        from api.main import process_floor_plan
    except ImportError:
        from .api.main import process_floor_plan
    result = await process_floor_plan(contents)
    return JSONResponse(content=result)

def get_gemini_api_key():
    # Check env var directly
    if os.environ.get("GEMINI_API_KEY"):
        return os.environ.get("GEMINI_API_KEY")
    if os.environ.get("VITE_API_KEY"):
        return os.environ.get("VITE_API_KEY")
    
    # Try reading from .env or ../.env
    for path in [".env", "../.env", "Backend/.env"]:
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    for line in f:
                        if line.startswith("VITE_API_KEY="):
                            return line.strip().split("=")[1]
                        if line.startswith("GEMINI_API_KEY="):
                            return line.strip().split("=")[1]
            except Exception as e:
                logger.warning(f"Error reading {path}: {e}")
    return None

class ChatMessage(BaseModel):
    role: str # 'user' or 'model'
    content: str

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = []
    floor_plan_context: Optional[dict] = None   # <-- analysis result passed from frontend

class ReportRequest(BaseModel):
    floor_plan_context: dict

# Chatbot endpoint - floor-plan aware expert advisor
@app.post("/chat")
async def chat(request: ChatRequest):
    api_key = get_gemini_api_key()
    if not api_key:
        raise HTTPException(status_code=500, detail="Gemini API Key not configured on the backend.")
    
    try:
        import google.generativeai as genai
        import warnings
        warnings.filterwarnings("ignore", category=FutureWarning)
        genai.configure(api_key=api_key)

        # Build context block if analysis data was provided
        context_block = ""
        if request.floor_plan_context:
            ctx = request.floor_plan_context
            rooms = ctx.get("rooms", [])
            metrics = ctx.get("metrics", {})
            suggestions = ctx.get("suggestions", [])
            room_summary = ", ".join(
                f"{r['type']} (area: {r.get('area', 'N/A')} px²)" for r in rooms
            ) or "No rooms detected"
            space_score = metrics.get("space_utilization", {}).get("space_score", "N/A")
            adj_score = metrics.get("adjacency", {}).get("adjacency_score", "N/A")
            acc_score = metrics.get("accessibility", {}).get("accessibility_score", "N/A")
            overall = metrics.get("overall_quality", "N/A")
            sugg_text = "\n".join(f"- {s}" for s in suggestions) or "None"
            context_block = (
                f"\n\n=== UPLOADED FLOOR PLAN ANALYSIS ===\n"
                f"Detected rooms: {room_summary}\n"
                f"Overall score: {overall}/100\n"
                f"Space utilization: {space_score}/100\n"
                f"Room adjacency: {adj_score}/100\n"
                f"Movement flow: {acc_score}/100\n"
                f"System suggestions:\n{sugg_text}\n"
                f"=== END OF ANALYSIS ===\n"
            )

        system_instruction = (
            "You are FloorGenie, a professional architectural floor plan advisor and interior design expert. "
            "Your PRIMARY job is to help users understand and improve their floor plans. "
            "When floor plan analysis data is provided, you MUST reference the actual rooms, scores, and issues in your advice. "
            "Be specific — mention the actual room types detected, the actual scores, and explain what they mean practically. "
            "Give concrete, actionable improvement suggestions (e.g., 'move kitchen next to dining room', "
            "'add a corridor connecting bedroom to bathroom', 'reduce wasted space in living room'). "
            "You also answer general questions about flooring materials, room dimensions, interior design, and space planning. "
            "If asked something completely unrelated to architecture, flooring, or interior design, politely redirect. "
            "Keep responses warm, professional, and easy to understand. Use bullet points for lists."
            + context_block
        )
        
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=system_instruction,
            generation_config={"temperature": 0.7}
        )
        
        # Format chat history
        contents = []
        if request.history:
            for msg in request.history:
                role = 'model' if msg.role in ['assistant', 'model'] else 'user'
                contents.append({"role": role, "parts": [msg.content]})
        
        contents.append({"role": "user", "parts": [request.message]})
        
        response = model.generate_content(contents)
        return {"reply": response.text}
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Gemini API error: {str(e)}")


# Report generation endpoint - writes a full diagnostic narrative
@app.post("/generate-report")
async def generate_report(request: ReportRequest):
    api_key = get_gemini_api_key()
    if not api_key:
        raise HTTPException(status_code=500, detail="Gemini API Key not configured.")
    
    try:
        import google.generativeai as genai
        import warnings
        warnings.filterwarnings("ignore", category=FutureWarning)
        genai.configure(api_key=api_key)

        ctx = request.floor_plan_context
        rooms = ctx.get("rooms", [])
        metrics = ctx.get("metrics", {})
        suggestions = ctx.get("suggestions", [])
        
        room_lines = "\n".join(
            f"- Room {r['id']+1}: {r['type']}, area={r.get('area','N/A')}px², "
            f"dimensions={r.get('bounding_box',{}).get('width','?')}x{r.get('bounding_box',{}).get('height','?')}"
            for r in rooms
        )
        space_score = metrics.get("space_utilization", {}).get("space_score", 0)
        adj_score = metrics.get("adjacency", {}).get("adjacency_score", 0)
        acc_score = metrics.get("accessibility", {}).get("accessibility_score", 0)
        overall = metrics.get("overall_quality", 0)
        sugg_text = "\n".join(f"- {s}" for s in suggestions)

        prompt = (
            "You are a senior architectural consultant writing a professional floor plan evaluation report.\n\n"
            f"FLOOR PLAN DATA:\n"
            f"Detected rooms:\n{room_lines}\n\n"
            f"Scores:\n"
            f"- Overall quality: {overall}/100\n"
            f"- Space utilization: {space_score}/100\n"
            f"- Room adjacency compatibility: {adj_score}/100\n"
            f"- Movement flow & accessibility: {acc_score}/100\n\n"
            f"System-generated suggestions:\n{sugg_text}\n\n"
            "Write a detailed but readable diagnostic report with these sections:\n"
            "1. EXECUTIVE SUMMARY - 2-3 sentences on overall quality and main verdict\n"
            "2. WHAT IS WRONG - Explain specifically WHY each score is low. "
            "   Be direct: 'The space utilization score of X is low because...', "
            "   'The adjacency score suffers because the kitchen is not near the dining room...', etc.\n"
            "3. ROOM-BY-ROOM ANALYSIS - For each room type detected, comment on its placement, size, and any issues.\n"
            "4. HOW TO IMPROVE - Specific, prioritized improvement steps (most impactful first).\n"
            "5. CONCLUSION - 1-2 sentences on potential after improvements.\n\n"
            "Write in professional but clear English. Use bold headings for sections. "
            "Do NOT use markdown code blocks. Plain text with **bold** for headings is fine."
        )

        model = genai.GenerativeModel(
            "gemini-2.5-flash",
            generation_config={"temperature": 0.4}
        )
        response = model.generate_content(prompt)
        return {"report": response.text}

    except Exception as e:
        logger.error(f"Report generation error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Report generation error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)