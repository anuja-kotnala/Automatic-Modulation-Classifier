import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

from api.routes import router as api_router

# Ensure output and results directories exist
os.makedirs("results", exist_ok=True)
os.makedirs(os.path.join("results", "plots"), exist_ok=True)
os.makedirs(os.path.join("results", "ml_eval"), exist_ok=True)
os.makedirs("dataset", exist_ok=True)

app = FastAPI(
    title="Automatic Modulation Classification (AMC) & Spectrum Analyzer API",
    description="REST backend for simulating baseband signals, applying impairments, extracting features, and classifying modulations.",
    version="1.0.0"
)

# CORS middleware configuration for easy React frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production environments
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve generated plots and evaluation assets statically
app.mount("/static", StaticFiles(directory="results"), name="static")

# Include the API router
app.include_router(api_router, tags=["API"])

from api.inference_dl import DLInferenceService

@app.on_event("startup")
def startup_event():
    DLInferenceService.load_all_models()

@app.get("/", include_in_schema=False)
def root_redirect():
    """Redirect root access to Swagger API documentation."""
    return RedirectResponse(url="/docs")

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
