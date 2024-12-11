import argparse
import os
import requests
import platform
import sys
from pathlib import Path
import cv2
import torch
import numpy as np
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                         increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode
from models.common import DetectMultiBackend

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

def calculate_ratio_score(head_to_body_ratio):
    print(f"\n등신 비율: {head_to_body_ratio:.2f} 등신")
    ideal_min = 7.0
    ideal_max = 10.0
    
    if ideal_min <= head_to_body_ratio <= ideal_max:
        score = 100.0
        print(f"등신 점수: {score:.1f}")
        print("적절한 등신 비율입니다.")
        feedback = "등신 분석\n적절한 등신 비율입니다."
    else:
        if head_to_body_ratio < ideal_min:
            difference = ideal_min - head_to_body_ratio
            penalty = min(difference * 20, 100)
            feedback = "등신 분석\n카메라 렌즈를 낮추면서 스마트폰을 피사체 반대로 눕히면 비율이 더 좋아보일 수 있습니다."
        else:
            difference = head_to_body_ratio - ideal_max
            penalty = min(difference * 20, 100)
            feedback = "등신 분석\n카메라 렌즈를 높이면서 스마트폰을 피사체 방향으로 눕히면 비율이 더 자연스러워 보일 수 있습니다."
        score = max(0, 100 - penalty)
        print(f"등신 점수: {score:.1f}")
        print(f"등신 피드백: {feedback}")
    
    return score, feedback

def calculate_height_ratio_score(person_height_ratio):
    print(f"\n전신/이미지 비율: {person_height_ratio:.1f}%")
    ideal_min = 65.0
    ideal_max = 85.0
    
    if ideal_min <= person_height_ratio <= ideal_max:
        score = 100.0
        print(f"전신 비율 점수: {score:.1f}")
        print("인물이 차지하는 비율이 적절합니다.")
        feedback = "비율 분석\n인물이 차지하는 비율이 적절합니다."
    else:
        if person_height_ratio < ideal_min:
            difference = ideal_min - person_height_ratio
            penalty = min(difference * 2, 100)
            feedback = "비율 분석\n인물이 차지하는 비율이 너무 적습니다. 조금 더 가까이에서 촬영해보세요. (65% ~ 85% 사이를 추천드립니다.)"
        else:
            difference = person_height_ratio - ideal_max
            penalty = min(difference * 2, 100)
            feedback = "비율 분석\n인물이 차지하는 비율이 너무 큽니다. 조금 더 멀리서 촬영해보세요. (65% ~ 85% 사이를 추천드립니다.)"
        score = max(0, 100 - penalty)
        print(f"전신 비율 점수: {score:.1f}")
        print(f"전신 비율 피드백: {feedback}")
    
    return score, feedback

def calculate_thirds_score(image_width, image_height, person_center_x):
    print("\n구도 분석")
    left_third = image_width / 3
    center = image_width / 2
    right_third = (image_width * 2) / 3
    
    min_distance = min(
        abs(person_center_x - left_third),
        abs(person_center_x - center),
        abs(person_center_x - right_third)
    )
    
    normalized_distance = min_distance / image_width
    score = 100 * (1 - normalized_distance)
    print(f"구도 점수: {score:.1f}")
    
    if normalized_distance < 0.05:
        print("인물이 3등분선 또는 중앙에 잘 위치되어 있습니다.")
        feedback = "구도 분석 \n인물이 3등분선 또는 중앙에 잘 위치되어 있습니다."
    elif normalized_distance < 0.15:
        feedback = "구도 분석 \n구도가 양호하나, 3등분선에 더 가깝게 위치하면 더 좋을 것 같습니다."
        print(f"구도 피드백: {feedback}")
    else:
        feedback = "구도 분석 \n인물을 화면의 1/3 지점이나 중앙에 위치시키면 더 안정적인 구도가 될 수 있습니다."
        print(f"구도 피드백: {feedback}")
    
    return score, feedback

def calculate_vertical_position_score(image_height, face_center_y, feet_y, head_y):
    print("\n수직 위치 분석")
    image_center_y = image_height / 2
    face_center_distance = abs(face_center_y - image_center_y)
    face_center_ratio = face_center_distance / image_height
    
    feet_distance = abs(feet_y - image_height)
    feet_ratio = feet_distance / image_height
    
    face_penalty = min(face_center_ratio * 200, 50)
    feet_penalty = min(feet_ratio * 200, 50)
    
    total_penalty = face_penalty + feet_penalty
    score = max(0, 100.0 - total_penalty)
    print(f"수직 위치 점수: {score:.1f}")
    
    if score >= 80:
        print("수직 위치 피드백: 인물의 수직 위치가 적절합니다")
        feedback = "수직 위치 분석\n인물의 수직 위치가 적절합니다.(얼굴은 중앙, 발은 프레임 아래)"
    elif feet_ratio > 0.1:
        feedback = "수직 위치 분석\n발이 프레임 아래쪽에 더 가깝게 위치하도록 구도를 잡아보세요"
        print(f"수직 위치 피드백: {feedback}")
    else:
        feedback = "수직 위치 분석\n얼굴이 화면 중앙에 오도록 구도를 조정해보세요"
        print(f"수직 위치 피드백: {feedback}")
    
    return score, feedback

def analyze_focus_difference(image, person_bbox, face_bbox):
    print("\n아웃포커싱 분석")
    def get_blur_score(roi):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    body_x1, body_y1, body_x2, body_y2 = person_bbox
    face_x, face_y, face_w, face_h = face_bbox
    
    face_roi = image[face_y:face_y+face_h, face_x:face_x+face_w]
    face_sharpness = get_blur_score(face_roi)
    
    mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
    mask[body_y1:body_y2, body_x1:body_x2] = 0
    
    background = cv2.bitwise_and(image, image, mask=mask)
    background_sharpness = get_blur_score(background)
    
    sharpness_diff = face_sharpness - background_sharpness
    max_diff = 1000

    # 수정된 점수 계산 로직
    normalized_diff = (sharpness_diff / max_diff) * 100
    
    if 30 <= normalized_diff <= 80:
        # 이상적인 아웃포커싱 범위
        base_score = 100.0
        # 중간값(55)에서 멀어질수록 점수 차감
        distance_from_optimal = abs(55 - normalized_diff)
        score = base_score - (distance_from_optimal * 0.5)  # 거리에 따라 완만하게 감소
        feedback = "아웃포커싱 분석\n아웃포커싱이 적절합니다."
    elif normalized_diff < 30:
        # 아웃포커싱이 부족한 경우
        score = 70 + (normalized_diff / 30) * 30  # 70~100점 사이 분포
        feedback = "아웃포커싱 분석\n아웃포커싱 효과를 조금 더 강화하면 좋겠습니다. 심도를 약간 더 높이거나 배경과의 거리를 더 확보해보세요."
    else:
        # 아웃포커싱이 과도한 경우
        score = 70 + ((100 - normalized_diff) / 20) * 30  # 70~100점 사이 분포
        feedback = "아웃포커싱 분석\n아웃포커싱이 약간 과하니 심도를 조금 낮추거나 배경과의 거리를 줄여보세요."

    score = max(70, min(100, score))  # 최소 70점, 최대 100점으로 제한
    print(f"아웃포커싱 점수: {score:.1f}")
    print(f"아웃포커싱 피드백: {feedback}")
    
    return score, feedback

def calculate_exposure(image, person_bbox, face_bbox):
    print("\n노출 분석")
    
    x1, y1, x2, y2 = person_bbox
    face_x, face_y, face_w, face_h = face_bbox
    
    # 마스크 생성
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255  # 인물 영역
    
    face_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    face_mask[face_y:face_y+face_h, face_x:face_x+face_w] = 255  # 얼굴 영역
    
    clothing_mask = cv2.bitwise_and(mask, cv2.bitwise_not(face_mask))  # 의상 영역
    
    # HSV 변환
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 명도(V) 분석
    v_channel = hsv_image[:, :, 2]
    face_brightness = np.mean(v_channel[face_mask == 255])
    clothing_brightness = np.mean(v_channel[clothing_mask == 255])
    background_brightness = np.mean(v_channel[mask == 0])
    
    # 점수 계산
    score = 100.0
    feedback_list = ["노출 분석"]
    
    print("\n노출 상세 분석:")
    
    # 얼굴 노출 분석
    if face_brightness < 50:
        score -= 30
        feedback = "얼굴이 너무 어둡습니다. 스마트폰 카메라 설정을 조정하거나 더 밝은 환경에서 촬영해보세요.\n"
        feedback += "조정 방법:\n"
        feedback += "1. 화면을 터치하여 얼굴에 초점을 맞춘 후, 밝기를 조정하세요.\n"
        feedback += "2. 더 밝은 조명 아래에서 촬영하세요.\n"
        feedback += "3. 플래시를 사용해보세요."
        feedback_list.append(feedback)
        print(feedback)
    elif face_brightness > 200:
        score -= 30
        feedback = "얼굴이 너무 밝습니다. 스마트폰 카메라 설정을 조정하거나 조명을 줄여보세요.\n"
        feedback += "조정 방법:\n"
        feedback += "1. 화면을 터치하여 얼굴에 초점을 맞춘 후, 밝기를 낮춰보세요.\n"
        feedback += "2. 더 어두운 장소에서 촬영하거나 조명을 줄이세요."
        feedback_list.append(feedback)
        print(feedback)
    else:
        print("얼굴의 노출이 적절합니다.")
        feedback_list.append("얼굴의 노출이 적절합니다.")
    
    # 의상 노출 분석
    if clothing_brightness < 50:
        score -= 20
        feedback = "의상이 너무 어둡습니다. 스마트폰 카메라 설정을 조정하거나 더 밝은 환경에서 촬영해보세요.\n"
        feedback += "조정 방법:\n"
        feedback += "1. 화면을 터치하여 의상에 초점을 맞춘 후, 밝기를 조정하세요.\n"
        feedback += "2. 더 밝은 조명 아래에서 촬영하세요.\n"
        feedback += "3. 플래시를 사용해보세요."
        feedback_list.append(feedback)
        print(feedback)
    elif clothing_brightness > 200:
        score -= 20
        feedback = "의상이 너무 밝습니다. 스마트폰 카메라 설정을 조정하거나 조명을 줄여보세요.\n"
        feedback += "조정 방법:\n"
        feedback += "1. 화면을 터치하여 의상에 초점을 맞춘 후, 밝기를 낮춰보세요.\n"
        feedback += "2. 더 어두운 장소에서 촬영하거나 조명을 줄이세요."
        feedback_list.append(feedback)
        print(feedback)
    else:
        print("의상의 노출이 적절합니다.")
        feedback_list.append("의상의 노출이 적절합니다.")
    
    # 배경 노출 분석
    if background_brightness < 50:
        score -= 10
        feedback = "배경이 너무 어둡습니다. 스마트폰 카메라 설정을 조정하거나 더 밝은 환경에서 촬영해보세요.\n"
        feedback += "조정 방법:\n"
        feedback += "1. 화면을 터치하여 배경에 초점을 맞춘 후, 밝기를 조정하세요.\n"
        feedback += "2. 더 밝은 장소에서 촬영하세요."
        feedback_list.append(feedback)
        print(feedback)
    elif background_brightness > 200:
        score -= 10
        feedback = "배경이 너무 밝습니다. 스마트폰 카메라 설정을 조정하거나 조명을 줄여보세요.\n"
        feedback += "조정 방법:\n"
        feedback += "1. 화면을 터치하여 배경에 초점을 맞춘 후, 밝기를 낮춰보세요.\n"
        feedback += "2. 어두운 장소에서 촬영하거나 조명을 줄이세요."
        feedback_list.append(feedback)
        print(feedback)
    else:
        print("배경의 노출이 적절합니다.")
        feedback_list.append("배경의 노출이 적절합니다.")
    
        # 최종 점수와 피드백
        score = max(0, score)
        print(f"\n노출 점수: {score:.1f}")
        
        if score == 100:
            final_feedback = "노출 분석\n전체적인 노출이 매우 적절합니다."
            print("\n노출 피드백: 전체적인 노출이 매우 적절합니다")
        else:
            final_feedback = "노출 분석\n" + "\n".join(feedback_list[1:])  # 첫 번째 "노출 분석" 제외
        
        return score, final_feedback

def calculate_color_balance(image, person_bbox, face_bbox):
    """
    색상 균형을 분석하는 함수
    
    Args:
        image: 원본 이미지
        person_bbox: 인물 바운딩 박스 (x1, y1, x2, y2)
        face_bbox: 얼굴 바운딩 박스 (x, y, w, h)
    
    Returns:
        (score, feedback) 튜플
    """
    print("\n색상 균형 분석")
    
    x1, y1, x2, y2 = person_bbox
    face_x, face_y, face_w, face_h = face_bbox
    
    # 마스크 생성
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    
    face_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    face_mask[face_y:face_y+face_h, face_x:face_x+face_w] = 255
    
    clothing_mask = cv2.bitwise_and(mask, cv2.bitwise_not(face_mask))

    # BGR 이미지를 HSV로 변환
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # V (명도) 채널에서 명도 계산
    v_channel = hsv_image[:, :, 2]
    face_brightness = np.mean(v_channel[face_mask == 255])  # 얼굴 영역의 명도 평균
    clothing_brightness = np.mean(v_channel[clothing_mask == 255])  # 의상 영역의 명도 평균
    background_brightness = np.mean(v_channel[mask == 0])  # 배경 영역의 명도 평
    # S (채도) 채널에서 채도 계산
    s_channel = hsv_image[:, :, 1]
    face_saturation = np.mean(s_channel[face_mask == 255])  # 얼굴 영역의 채도 평균
    clothing_saturation = np.mean(s_channel[clothing_mask == 255])  # 의상 영역의 채도 평균
    background_saturation = np.mean(s_channel[mask == 0])  # 배경 영역의 채도 평
    # 결과 출력
    print(f"Face brightness: {face_brightness}")
    print(f"Clothing brightness: {clothing_brightness}")
    print(f"Background brightness: {background_brightness}")
    print(f"Face saturation: {face_saturation}")
    print(f"Clothing saturation: {clothing_saturation}")
    print(f"Background saturation: {background_saturation}")
    # 얼굴과 의상 간의 명도 차이 계산 (0~255 범위)
    face_clothing_diff = abs(face_brightness - clothing_brightness)
    face_bg_diff = abs(face_brightness - background_brightness)
    clothing_bg_diff = abs(clothing_brightness - background_brightness)
    # 얼굴과 의상 간의 채도 차이 계산 (0~255 범위)
    face_clothing_saturation_diff = abs(face_saturation - clothing_saturation)
    face_bg_saturation_diff = abs(face_saturation - background_saturation)
    clothing_bg_saturation_diff = abs(clothing_saturation - background_saturation)
    # 명도 차이를 퍼센트로 변환
    face_clothing_diff_pct = (face_clothing_diff / 255) * 100
    face_bg_diff_pct = (face_bg_diff / 255) * 100
    clothing_bg_diff_pct = (clothing_bg_diff / 255) * 10
    # 채도 차이를 퍼센트로 변환
    face_clothing_saturation_diff_pct = (face_clothing_saturation_diff / 255) * 100
    face_bg_saturation_diff_pct = (face_bg_saturation_diff / 255) * 100
    clothing_bg_saturation_diff_pct = (clothing_bg_saturation_diff / 255) * 10
    print(f"Face-Clothing brightness difference: {face_clothing_diff_pct:.2f}%")
    print(f"Face-Background brightness difference: {face_bg_diff_pct:.2f}%")
    print(f"Clothing-Background brightness difference: {clothing_bg_diff_pct:.2f}%")
    print(f"Face-Clothing saturation difference: {face_clothing_saturation_diff_pct:.2f}%")
    print(f"Face-Background saturation difference: {face_bg_saturation_diff_pct:.2f}%")
    print(f"Clothing-Background saturation difference: {clothing_bg_saturation_diff_pct:.2f}%")
    
    # 점수 계산
    score = 100.0
    feedbacks = ["명도 채도 분석"]  # feedbacks 리스트 초기화
    
    # 얼굴과 의상의 명도 차이 피드백
    if face_clothing_diff_pct < 5:
        feedbacks.append("얼굴과 의상의 명도 차이가 매우 적어, 얼굴이 잘 부각되지 않을 수 있습니다. 의상을 조금 더 밝은 색으로 선택하거나, 얼굴의 명도를 조금 더 높이면 좋습니다. 화이트 밸런스를 조정하거나 노출 보정을 통해 얼굴의 밝기를 높여 보세요. 또한, 얼굴의 명도를 높이는 방법으로 하이라이트를 조정하거나, 디지털 편집 프로그램에서 'Brightness/Contrast' 기능을 사용해보는 것도 좋은 방법입니다.")
    elif 5 <= face_clothing_diff_pct <= 15:
        feedbacks.append("얼굴과 의상의 명도 차이가 적당하여 얼굴이 자연스럽게 부각됩니다. 이 상태는 매우 자연스럽습니다. 하지만, 'Vibrance' 슬라이더를 사용해 얼굴의 색감을 살리거나, 의상의 채도를 살짝 조정하여 좀 더 선명하고 생동감 있는 효과를 줄 수 있습니다.")
    elif face_clothing_diff_pct > 15:
        feedbacks.append("얼굴과 의상의 명도 차이가 커서 얼굴이 잘 부각됩니다. 그러나 너무 강조될 수 있으므로 의상의 명도를 조금 낮추거나, 얼굴의 위치를 조금 바꿔서 자연스럽게 만들 수 있습니다. 채도를 조정하여 의상의 색이 너무 강하지 않게 할 수 있습니다. 또한, 'Curves' 도구를 사용하여 의상과 얼굴의 명도를 더 부드럽게 맞춰 자연스러운 효과를 얻을 수 있습니다.")

    # 의상과 배경의 명도 차이 피드백
    if clothing_bg_diff_pct < 10:
        feedbacks.append("의상과 배경의 명도 차이가 적어, 피사체가 배경에 자연스럽게 녹아듭니다. 의상 색상을 더 밝은 색으로 선택하거나 배경의 명도를 낮춰 보세요. 'Exposure Compensation' 기능을 사용하여 배경의 밝기를 낮추거나, 의상에 채도를 높여 대비를 강화할 수 있습니다.")
    elif 10 <= clothing_bg_diff_pct <= 30:
        feedbacks.append("의상과 배경의 명도 차이가 적당하여 피사체가 잘 부각됩니다. 현재 상태는 잘 조화된 이미지입니다. 하지만, 배경의 색상을 조금 더 차분하게 만들거나, 의상의 채도를 높여서 색감의 강렬함을 조절하는 방법을 고려할 수 있습니다.")
    elif clothing_bg_diff_pct > 30:
        feedbacks.append("의상과 배경의 명도 차이가 커서 피사체가 배경에서 과도하게 강조됩니다. 의상 색상을 조금 더 어두운 톤으로 선택하거나 배경을 흐리게 하여 조화롭게 만들 수 있습니다. 배경을 흐리게 만들기 위해 'Blur' 효과를 사용하거나, 의상 색상을 좀 더 톤 다운하여 자연스러운 조화를 이룰 수 있습니다. 또한, Vibrance 기능으로 색감을 조금 더 부드럽게 할 수 있습니다.")
    
    # 얼굴과 의상의 채도 차이 피드백
    if face_clothing_saturation_diff_pct < 5:
        feedbacks.append("얼굴과 의상의 채도가 비슷하여 색감이 너무 단조로워 보일 수 있습니다. 의상을 조금 더 채도가 높은 색으로 선택하거나 얼굴에 약간의 채도를 추가해 보세요. 채도를 높이기 위해 'Saturation' 슬라이더를 오른쪽으로 이동시키거나, 'Vibrance'로 자연스럽게 색감을 강조할 수 있습니다.")
    elif 5 <= face_clothing_saturation_diff_pct <= 15:
        feedbacks.append("얼굴과 의상의 채도 차이가 적당하여 색감이 자연스럽게 맞아떨어집니다. 이 상태는 매우 균형 잡힌 이미지입니다. 그러나 배경과 의상의 색상을 대비시켜 강렬한 이미지를 만들고 싶다면 의상의 채도를 조금 더 높여 보세요.")
    elif face_clothing_saturation_diff_pct > 15:
        feedbacks.append("얼굴과 의상의 채도 차이가 커서 색이 많이 강조됩니다. 배경의 색감을 살짝 조정하여 전체적인 색 균형을 맞추거나, 의상 색상을 조금 더 자연스럽게 만들기 위해 'Saturation' 슬라이더를 낮추는 것도 좋습니다. 또한, 의상과 배경의 색조가 너무 겹치지 않도록 색감을 조정하여 균형을 맞출 수 있습니다.")

    # 의상과 배경의 채도 차이 피드백
    if clothing_bg_saturation_diff_pct < 5:
        feedbacks.append("의상과 배경의 채도가 비슷하여 색감이 너무 평범하게 느껴질 수 있습니다. 의상 색감을 좀 더 뚜렷하게 강조하거나, 배경을 흐리게 하여 더 두드러지게 만들 수 있습니다. 또한, 배경의 채도를 조금 낮추어 의상이 더욱 눈에 띄게 할 수 있습니다.")
    elif 5 <= clothing_bg_saturation_diff_pct <= 15:
        feedbacks.append("의상과 배경의 채도 차이가 적당하여 색감이 자연스럽게 어우러집니다. 현재 상태는 균형 잡힌 이미지입니다. 배경의 채도를 조절하거나, 의상의 색상을 더 진하게 만들어 색감의 대비를 더 강하게 만들 수 있습니다.")
    elif clothing_bg_saturation_diff_pct > 15:
        feedbacks.append("의상과 배경의 채도 차이가 커서 색이 많이 강조됩니다. 이 경우 배경의 채도를 조금 낮추거나, 의상의 색감을 더 자연스럽게 만들어 조화를 이룰 수 있습니다.")

    # 피드백 결합
    if len(feedbacks) > 1:
        final_feedback = feedbacks[0] + "\n" + "\n".join([fb for fb in feedbacks[1:] if fb])  # 첫 번째 "명도 채도 분석" 유지
    else:
        final_feedback = "명도 채도 분석\n전체적인 색감이 적절합니다."
    
    # 피드백 출력
    print("\n피드백:")
    print(final_feedback)
    
    return score, final_feedback

def combine_evaluation_results(measurements: dict, scores_and_feedbacks: list) -> dict:
    """
    평가 결과들을 합치고 최종 형식으로 변환하는 함수
    
    Args:
        measurements: 측정값들을 담은 딕셔너리
        scores_and_feedbacks: (score, feedback) 튜플들의 리스트
    
    Returns:
        API 응답 형식에 맞는 최종 결과 딕셔너리
    """
    # 가중치 설정
    weights = {
        'ratio': 0.2,       # 등신 비율
        'height': 0.2,     # 전신 비율
        'composition': 0.2,  # 구도
        'vertical': 0.1,   # 수직 위치
        'focus': 0.1,      # 아웃포커싱
        'exposure': 0.1,        # 노출
        'color_balance': 0.1  # 색상 균형
    }
    
    # 점수 계산
    scores = {}
    feedbacks = []
    feedback_id = 1
    
    for category, (score, feedback) in zip(weights.keys(), scores_and_feedbacks):
        scores[category] = score
        if feedback is not None:
            feedbacks.append([feedback_id, feedback])
            feedback_id += 1
    
    # 총점 계산
    total_score = sum(score * weights[category] 
                     for category, score in scores.items())

    # 총점 출력
    print("\n=== 최종 평가 결과 ===")
    print(f"총점: {total_score:.1f}/100")
    
    # 측정값들을 문자열로 변환
    more_info = ", ".join([f"{k}: {v}" for k, v in measurements.items()])
    
    # 최종 결과 형식 구성
    final_result = {
        "totalScore": round(total_score, 1),
        "feedback": feedbacks,
        "moreInfo": more_info
    }
    
    return final_result

def process_image(image_url):
    """
    이미지 URL을 받아서 평가를 수행하는 함수
    
    Args:
        image_url: 평가할 이미지의 URL
    
    Returns:
        평가 결과 딕셔너리
    """
    try:
        # 이미지 다운로드
        response = requests.get(image_url)
        response.raise_for_status()
        
        # 임시 파일로 저장
        temp_path = "temp_image.jpg"
        with open(temp_path, "wb") as f:
            f.write(response.content)

        # YOLO 모델로 이미지 분석 실행
        result = run(
            weights='yolov5s.pt',
            source=temp_path,
            save_txt=False,
            save_conf=False,
            save_crop=False,
            nosave=True
        )

        # 임시 파일 삭제
        if os.path.exists(temp_path):
            os.remove(temp_path)

        return result

    except Exception as e:
        return {
            "totalScore": 0,
            "feedback": [[1, f"이미지 처리 중 오류가 발생했습니다: {str(e)}"]],
            "moreInfo": "처리 실패"
        }

def resize_image(image, max_size=1280):
    """
    이미지 크기가 너무 큰 경우 리사이즈하는 함수
    
    Args:
        image: 원본 이미지 (numpy array)
        max_size: 최대 허용 크기 (픽셀)
    
    Returns:
        리사이즈된 이미지
    """
    # 이미지의 현재 크기 확인
    height, width = image.shape[:2]
    
    # 현재 크기 출력
    print(f"\n원본 이미지 크기: {width}x{height}")
    
    # 최대 크기를 초과하는지 확인
    if max(height, width) > max_size:
        # 리사이징 비율 계산
        ratio = max_size / float(max(height, width))
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        
        # 이미지 리사이징
        resized = cv2.resize(image, (new_width, new_height), 
                           interpolation=cv2.INTER_AREA)
        
        print(f"리사이즈된 이미지 크기: {new_width}x{new_height}")
        return resized
    
    print("이미지 크기가 적절하여 리사이징이 필요하지 않습니다.")
    return image

def process_single_detection(image, det, face_detection):
    """
    단일 인물 검출 결과를 처리하는 함수
    
    Args:
        image: 원본 이미지
        det: YOLO 검출 결과
        face_detection: 얼굴 검출 결과
    
    Returns:
        평가 결과 딕셔너리
    """

    # 이미지 리사이징 추가
    image = resize_image(image)

    # 기본 측정값 계산
    body_x1, body_y1, body_x2, body_y2 = [int(x) for x in det[:4]]
    body_height = body_y2 - body_y1
    person_center_x = (body_x1 + body_x2) / 2
    image_height = image.shape[0]
    image_width = image.shape[1]
    person_height_ratio = (body_height / image_height) * 100
    
    # 얼굴 관련 측정값 계산
    x, y, w, h = face_detection
    face_height = h
    head_to_body_ratio = body_height / face_height
    face_center_y = y + (h / 2)
    feet_y = body_y2
    head_y = body_y1

    # 아웃포커싱 측정값 계산
    def get_blur_score(roi):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    face_roi = image[y:y+h, x:x+w]
    face_sharpness = get_blur_score(face_roi)
    
    mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
    mask[body_y1:body_y2, body_x1:body_x2] = 0
    background = cv2.bitwise_and(image, image, mask=mask)
    background_sharpness = get_blur_score(background)
    
    # 명도와 채도 관련 값 계산
    x1, y1, x2, y2 = [int(x) for x in det[:4]]
    x, y, w, h = face_detection
    
    # 마스크 생성
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    
    face_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    face_mask[y:y+h, x:x+w] = 255
    
    clothing_mask = cv2.bitwise_and(mask, cv2.bitwise_not(face_mask))
    
    # HSV 변환 및 분석
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v_channel = hsv_image[:, :, 2]
    s_channel = hsv_image[:, :, 1]
    
    # 명도와 채도 계산
    face_brightness = np.mean(v_channel[face_mask == 255])
    clothing_brightness = np.mean(v_channel[clothing_mask == 255])
    background_brightness = np.mean(v_channel[mask == 0])
    
    face_saturation = np.mean(s_channel[face_mask == 255])
    clothing_saturation = np.mean(s_channel[clothing_mask == 255])
    background_saturation = np.mean(s_channel[mask == 0])
    
    # 차이값 계산
    face_clothing_diff = abs(face_brightness - clothing_brightness)
    face_bg_diff = abs(face_brightness - background_brightness)
    clothing_bg_diff = abs(clothing_brightness - background_brightness)
    
    face_clothing_saturation_diff = abs(face_saturation - clothing_saturation)
    face_bg_saturation_diff = abs(face_saturation - background_saturation)
    clothing_bg_saturation_diff = abs(clothing_saturation - background_saturation)
    
    # 퍼센트로 변환
    face_clothing_diff_pct = (face_clothing_diff / 255) * 100
    face_bg_diff_pct = (face_bg_diff / 255) * 100
    clothing_bg_diff_pct = (clothing_bg_diff / 255) * 100
    
    face_clothing_saturation_diff_pct = (face_clothing_saturation_diff / 255) * 100
    face_bg_saturation_diff_pct = (face_bg_saturation_diff / 255) * 100
    clothing_bg_saturation_diff_pct = (clothing_bg_saturation_diff / 255) * 100

    
    measurements = {
        "전신 세로 길이": f"{body_height} pixels",
        "얼굴 세로 길이": f"{face_height} pixels",
        "등신 비율": f"{head_to_body_ratio:.2f} 등신",
        "전신/이미지 비율": f"{person_height_ratio:.1f}%",
        "얼굴 선명도": f"{face_sharpness:.1f}",
        "배경 선명도": f"{background_sharpness:.1f}",
        "선명도 차이": f"{face_sharpness - background_sharpness:.1f}",
        # 명도 관련 측정값
        "얼굴 밝기": f"{face_brightness:.1f}",
        "의상 밝기": f"{clothing_brightness:.1f}",
        "배경 밝기": f"{background_brightness:.1f}",
        "얼굴-의상 밝기차": f"{face_clothing_diff_pct:.1f}%",
        "얼굴-배경 밝기차": f"{face_bg_diff_pct:.1f}%",
        "의상-배경 밝기차": f"{clothing_bg_diff_pct:.1f}%",
        # 채도 관련 측정값
        "얼굴 채도": f"{face_saturation:.1f}",
        "의상 채도": f"{clothing_saturation:.1f}",
        "배경 채도": f"{background_saturation:.1f}",
        "얼굴-의상 채도차": f"{face_clothing_saturation_diff_pct:.1f}%",
        "얼굴-배경 채도차": f"{face_bg_saturation_diff_pct:.1f}%",
        "의상-배경 채도차": f"{clothing_bg_saturation_diff_pct:.1f}%"
    }
    
    # 각 평가 수행
    scores_and_feedbacks = [
        calculate_ratio_score(head_to_body_ratio),
        calculate_height_ratio_score(person_height_ratio),
        calculate_thirds_score(image_width, image_height, person_center_x),
        calculate_vertical_position_score(image_height, face_center_y, feet_y, head_y),
        analyze_focus_difference(image, (body_x1, body_y1, body_x2, body_y2), (x, y, w, h)),
        calculate_exposure(image, (body_x1, body_y1, body_x2, body_y2), (x, y, w, h)),
        calculate_color_balance(image, (body_x1, body_y1, body_x2, body_y2), (x, y, w, h))
    ]
    
    # 결과 합치기
    return combine_evaluation_results(measurements, scores_and_feedbacks)

@smart_inference_mode()
def run(
    weights=ROOT / 'yolov5s.pt',
    source=ROOT / 'data/images',
    data=ROOT / 'data/coco128.yaml',
    imgsz=(640, 640),
    conf_thres=0.25,
    iou_thres=0.45,
    max_det=1000,
    device='',
    view_img=False,
    save_txt=False,
    save_conf=False,
    save_crop=False,
    nosave=False,
    classes=None,
    agnostic_nms=False,
    augment=False,
    visualize=False,
    update=False,
    project=ROOT / 'runs/detect',
    name='exp',
    exist_ok=False,
    line_thickness=3,
    hide_labels=False,
    hide_conf=False,
    half=False,
    dnn=False,
    vid_stride=1,
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    # Load face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Dataloader
    bs = 1
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    
    for path, im, im0s, vid_cap, s in dataset:
        # 이미지 로드 후 바로 리사이징 적용
        if isinstance(im0s, np.ndarray):
            im0s = resize_image(im0s)
        elif isinstance(im0s, list):
            im0s = [resize_image(img) for img in im0s]

        with dt[0]:
            im = torch.from_numpy(im).to(device)
            im = im.half() if model.fp16 else im.float()
            im /= 255
            if len(im.shape) == 3:
                im = im[None]

        # Inference
        with dt[1]:
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Process predictions
        for i, det in enumerate(pred):
            seen += 1
            if webcam:
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            if len(det):
                # Rescale boxes from img_size to im0 size
                det = scale_boxes(im.shape[2:], det, im0.shape).round()

                # 사람 객체만 필터링 (클래스 0이 person)
                person_dets = det[det[:, -1] == 0]  # person class = 0

                if len(person_dets) > 1:
                    print(f"\n경고: {len(person_dets)}명의 사람이 감지되었습니다. 가장 큰 영역의 인물을 분석합니다.")
                    # CUDA 텐서를 CPU로 옮기고 NumPy로 변환
                    person_dets_cpu = person_dets.cpu().numpy()
                    areas = (person_dets_cpu[:, 2] - person_dets_cpu[:, 0]) * (person_dets_cpu[:, 3] - person_dets_cpu[:, 1])
                    main_person_idx = areas.argmax()
                    person_det = person_dets_cpu[main_person_idx]
                elif len(person_dets) == 1:
                    person_det = person_dets[0].cpu().numpy()
                else:
                    continue
                
                # Face detection
                gray = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                
                if len(faces) > 1:
                    print(f"\n경고: {len(faces)}개의 얼굴이 감지되었습니다. 주요 인물의 상단과 가장 가까운 얼굴을 선택합니다.")
                    
                    # 주요 인물의 상단 y좌표
                    person_top = person_det[1]  # y1 좌표
                    
                    # 각 얼굴의 하단 y좌표와 주요 인물 상단과의 거리 계산
                    distances = []
                    for (x, y, w, h) in faces:
                        face_bottom = y + h  # 얼굴 바운딩 박스의 하단 y좌표
                        # 수직 거리만 계산
                        dist = abs(face_bottom - person_top)
                        distances.append(dist)
                    
                    # 가장 가까운 얼굴 선택
                    main_face_idx = np.argmin(distances)
                    main_face = faces[main_face_idx]
                    
                    # 선택된 얼굴의 정보 출력
                    x, y, w, h = main_face
                    print(f"선택된 얼굴 위치: 상단({y}), 하단({y+h}), 주요 인물 상단({person_top})")
                elif len(faces) == 1:
                    main_face = faces[0]
                else:
                    print("\n경고: 얼굴이 감지되지 않았습니다.")
                    continue

                # 선택된 인물과 얼굴로 분석 진행
                result = process_single_detection(im0, person_det, main_face)
                
                if view_img:
                    # 시각화 코드
                    annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                    # 선택된 인물 박스 그리기
                    c = int(person_det[-1])
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {person_det[4]:.2f}')
                    annotator.box_label(person_det[:4], label, color=colors(c, True))
                    # 선택된 얼굴 박스 그리기
                    x, y, w, h = main_face
                    cv2.rectangle(im0, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    
                    # 결과 표시
                    if platform.system() == 'Linux' and p not in windows:
                        windows.append(p)
                        cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                        cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(10000)

                return result

    # 인물이 감지되지 않은 경우
    return {
        "totalScore": 0,
        "feedback": [[1, "인물이 감지되지 않았습니다. 전신이 모두 나오도록 촬영해주세요."]],
        "moreInfo": "인물 감지 실패"
    }

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt

def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)