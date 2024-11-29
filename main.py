from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from detect import process_image  # detect.py에서 이미지 처리 함수 import

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImageEvaluationRequest(BaseModel):
    image_url: str
    image_evaluation_id: int

class ImageEvaluationRequest(BaseModel):
    image_url: str
    image_evaluation_id: int

    class Config:
        schema_extra = {
            "example": {
                "image_url": "http://example.com/image.jpg",
                "image_evaluation_id": 1
            }
        }

@app.post("/evaluate_image")
async def evaluate_image_endpoint(request: ImageEvaluationRequest):
    try:
        result = process_image(request.image_url)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)