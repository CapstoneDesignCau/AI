# main.py ###
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

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