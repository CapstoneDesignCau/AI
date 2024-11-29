from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용
    allow_headers=["*"],  # 모든 HTTP 헤더 허용
)

class RatioRequest(BaseModel):
    head_to_body_ratio: float

class VerticalPositionRequest(BaseModel):
    image_height: int
    face_center_y: int
    feet_y: int
    head_y: int

@app.post("/calculate_ratio_score")
def calculate_ratio_score_endpoint(request: RatioRequest):
    score = calculate_ratio_score(request.head_to_body_ratio)
    return {"score": score}

@app.post("/calculate_vertical_position_score")
def calculate_vertical_position_score_endpoint(request: VerticalPositionRequest):
    score = calculate_vertical_position_score(request.image_height, request.face_center_y, request.feet_y, request.head_y)
    return {"score": score}

def calculate_ratio_score(head_to_body_ratio):
    ideal_min = 7.0
    ideal_max = 10.0
    
    if ideal_min <= head_to_body_ratio <= ideal_max:
        return 100.0
    
    if head_to_body_ratio < ideal_min:
        difference = ideal_min - head_to_body_ratio
        penalty = min(difference * 20, 100)
    else:
        difference = head_to_body_ratio - ideal_max
        penalty = min(difference * 20, 100)
    
    score = max(0, 100 - penalty)
    return score

def calculate_vertical_position_score(image_height, face_center_y, feet_y, head_y):
    score = 100.0
    penalties = []
    
    image_center_y = image_height / 2
    
    # Add your logic here
    
    return score

class ImageEvaluationRequest(BaseModel):
    image_url: str
    image_evaluation_id: int

@app.post("/evaluate_image")
def evaluate_image_endpoint(request: ImageEvaluationRequest):
    print(request)
    response = {
        "totalScore": 90,
        "feedback": [
            [1, "아웃포커싱 효과를 활용하면 인물을 더 돋보이게 할 수 있습니다."],
            [2, "카메라를 아래쪽에서 찍어 보세요."],
            [3, "사진이 너무 어둡습니다. 명도를 올려보세요."]
        ],
        "moreInfo": "전신 세로 길이: 25 pixels, 얼굴 세로 길이: 10 pixels, 등신 비율: 7 등신"
    }
    return response
