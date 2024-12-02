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
        print("등신 피드백: 적절한 등신 비율입니다")
        feedback = None
    else:
        if head_to_body_ratio < ideal_min:
            difference = ideal_min - head_to_body_ratio
            penalty = min(difference * 20, 100)
            feedback = "카메라를 낮추어 촬영하면 다리가 더 길어 보일 수 있습니다"
        else:
            difference = head_to_body_ratio - ideal_max
            penalty = min(difference * 20, 100)
            feedback = "카메라를 높여서 촬영하면 전신 비율이 더 자연스러워질 수 있습니다"
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
        print("전신 비율 피드백: 인물이 프레임에 잘 맞게 촬영되었습니다")
        feedback = None
    else:
        if person_height_ratio < ideal_min:
            difference = ideal_min - person_height_ratio
            penalty = min(difference * 2, 100)
            feedback = "인물을 좀 더 크게 촬영하면 좋겠습니다. 카메라를 더 가까이 가져가보세요"
        else:
            difference = person_height_ratio - ideal_max
            penalty = min(difference * 2, 100)
            feedback = "인물이 프레임에 비해 너무 큽니다. 카메라를 조금 더 멀리서 촬영해보세요"
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
        print("구도 피드백: 인물이 3등분선 위치에 잘 배치되어 있습니다")
        feedback = None
    elif normalized_distance < 0.15:
        feedback = "구도가 양호하나, 3등분선에 더 가깝게 위치하면 더 좋을 것 같습니다"
        print(f"구도 피드백: {feedback}")
    else:
        feedback = "인물을 화면의 1/3 지점이나 중앙에 위치시키면 더 안정적인 구도가 될 수 있습니다"
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
        feedback = None
    elif feet_ratio > 0.1:
        feedback = "발이 프레임 아래쪽에 더 가깝게 위치하도록 구도를 잡아보세요"
        print(f"수직 위치 피드백: {feedback}")
    else:
        feedback = "얼굴이 화면 중앙에 오도록 구도를 조정해보세요"
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
    score = min(100, max(0, (sharpness_diff / max_diff) * 100))
    print(f"아웃포커싱 점수: {score:.1f}")
    
    if score < 30:
        feedback = "아웃포커싱 효과를 활용하면 인물을 더 돋보이게 할 수 있습니다"
        print(f"아웃포커싱 피드백: {feedback}")
    elif score > 80:
        feedback = "배경의 아웃포커싱이 과도합니다. 조금 더 줄여보세요"
        print(f"아웃포커싱 피드백: {feedback}")
    else:
        print("아웃포커싱 피드백: 아웃포커싱이 적절합니다")
        feedback = None
    
    return score, feedback

def calculate_f_value(sensor_type, focal_length, subject_distance, desired_dof):
    """
    F값 계산 함수
    :param sensor_type: 센서 유형 ('full_frame', 'aps_c', 'micro_4_3' 등)
    :param focal_length: 초점 거리 (mm)
    :param subject_distance: 피사체와 카메라 간 거리 (m)
    :param desired_dof: 원하는 심도 (m)
    :return: 추천 F값
    """
    # 허용 혼합 원(c) 설정
    circle_of_confusion = {
        "full_frame": 0.03,
        "aps_c": 0.02,
        "micro_4_3": 0.015
    }
    c = circle_of_confusion.get(sensor_type, 0.03)  # 기본 풀프레임 기준

    try:
        f_value = (2 * c * subject_distance**2) / (desired_dof * focal_length**2)
        return max(1.4, round(f_value, 1))  # F값은 최소 1.4로 설정
    except ZeroDivisionError:
        return None

def evaluate_magnifications(image, max_zoom=10, sensor_type="full_frame", base_focal_length=26, base_distance=1, desired_dof=0.1):
    """
    배율(줌)별 적합성 평가 함수
    :param image: 입력 이미지
    :param max_zoom: 최대 배율
    :param sensor_type: 카메라 센서 유형
    :param base_focal_length: 1배율의 초점 거리 (mm)
    :param base_distance: 기본 피사체 거리 (m)
    :param desired_dof: 원하는 심도
    :return: 추천 배율 및 관련 정보
    """
    best_magnification = None
    best_score = float("-inf")
    recommendations = []

    for zoom in range(1, max_zoom + 1):
        # 배율에 따른 초점 거리 및 피사체 거리 설정
        focal_length = base_focal_length * zoom  # 초점 거리 = 기본 초점 거리 × 배율
        subject_distance = base_distance * zoom  # 피사체 거리도 배율에 비례

        # F값 계산
        f_value = calculate_f_value(sensor_type, focal_length, subject_distance, desired_dof)

        # 배율에 따라 이미지를 시뮬레이션(ROI 설정)
        height, width = image.shape[:2]
        x_center, y_center = width // 2, height // 2
        zoomed_image = image[
            max(0, y_center - height // (2 * zoom)):min(height, y_center + height // (2 * zoom)),
            max(0, x_center - width // (2 * zoom)):min(width, x_center + width // (2 * zoom))
        ]

        # BGR 이미지를 HSV로 변환
        hsv_image = cv2.cvtColor(zoomed_image, cv2.COLOR_BGR2HSV)

        # V 채널에서 밝기 계산
        v_channel = hsv_image[:, :, 2]
        person_brightness = np.mean(v_channel)
        background_brightness = np.mean(v_channel)
        brightness_difference = abs(person_brightness - background_brightness)

        # 점수 계산: 밝기 차이와 F값 고려
        score = 0
        if 10 <= brightness_difference <= 30:
            score += 20  # 이상적인 밝기 차이 범위
        score -= abs(f_value - 5)  # F값에서 이상적인 값(5)과의 차이를 페널티로 적용

        recommendations.append((zoom, focal_length, f_value, score))
        if score > best_score:
            best_score = score
            best_magnification = zoom

    return best_magnification, recommendations

def calculate_magnification_score(image, person_bbox):
    """
    배율 적합성을 평가하는 함수
    
    Args:
        image: 원본 이미지
        person_bbox: 인물 바운딩 박스
    
    Returns:
        (score, feedback) 튜플
    """
    print("\n배율 분석")
    
    # 기본 파라미터 설정
    sensor_type = "full_frame"
    base_focal_length = 26
    base_distance = 1
    desired_dof = 0.1
    
    # 배율 평가 실행
    best_magnification, recommendations = evaluate_magnifications(
        image, 
        max_zoom=10,
        sensor_type=sensor_type,
        base_focal_length=base_focal_length,
        base_distance=base_distance,
        desired_dof=desired_dof
    )
    
    # 최적 추천값 찾기
    best_rec = next((rec for rec in recommendations if rec[0] == best_magnification), None)
    if best_rec:
        zoom, focal_length, f_value, score = best_rec
        normalized_score = min(100, max(0, score + 80))  # 점수 정규화
        
        print(f"배율 점수: {normalized_score:.1f}")
        
        if normalized_score >= 80:
            print("배율 피드백: 현재 배율이 적절합니다")
            feedback = None
        else:
            feedback = f"추천 배율: {zoom}배 (f/{f_value:.1f}, 초점거리: {focal_length}mm)"
            print(f"배율 피드백: {feedback}")
        
        return normalized_score, feedback
    else:
        return 0, "배율을 평가할 수 없습니다"        

def calculate_exposure(image, person_bbox, face_bbox):
    """
    노출과 채도를 분석하는 함수
    
    Args:
        image: 원본 이미지
        person_bbox: 인물 바운딩 박스 (x1, y1, x2, y2)
        face_bbox: 얼굴 바운딩 박스 (x, y, w, h)
    
    Returns:
        (score, feedback) 튜플
    """
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
    
    # 채도(S) 분석
    s_channel = hsv_image[:, :, 1]
    face_saturation = np.mean(s_channel[face_mask == 255])
    clothing_saturation = np.mean(s_channel[clothing_mask == 255])
    background_saturation = np.mean(s_channel[mask == 0])
    
    # 점수 계산
    score = 100.0
    feedback_list = []
    
    # 얼굴 노출 평가
    if face_brightness < 50:
        score -= 30
        feedback_list.append("얼굴이 너무 어둡습니다. 노출을 늘리거나 조명을 밝게 하세요")
    elif face_brightness > 200:
        score -= 30
        feedback_list.append("얼굴이 너무 밝습니다. 노출을 줄이거나 조명을 어둡게 하세요")
    
    # 의상 노출 평가
    if clothing_brightness < 50:
        score -= 20
        feedback_list.append("의상이 너무 어둡습니다")
    elif clothing_brightness > 200:
        score -= 20
        feedback_list.append("의상이 너무 밝습니다")
    
    # 배경 노출 평가
    if background_brightness < 50:
        score -= 10
        feedback_list.append("배경이 너무 어둡습니다")
    elif background_brightness > 200:
        score -= 10
        feedback_list.append("배경이 너무 밝습니다")
    
    # 최종 점수와 피드백
    score = max(0, score)
    print(f"노출 점수: {score:.1f}")
    
    if feedback_list:
        feedback = " / ".join(feedback_list)
        print(f"노출 피드백: {feedback}")
    else:
        feedback = None
        print("노출 피드백: 노출이 전체적으로 적절합니다")
    
    return score, feedback

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
    
    # HSV 변환 및 분석
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    s_channel = hsv_image[:, :, 1]
    v_channel = hsv_image[:, :, 2]
    
    # 채도와 명도 계산
    face_saturation = np.mean(s_channel[face_mask == 255])
    clothing_saturation = np.mean(s_channel[clothing_mask == 255])
    background_saturation = np.mean(s_channel[mask == 0])
    
    # 차이값 계산 및 퍼센트 변환
    face_clothing_saturation_diff_pct = (abs(face_saturation - clothing_saturation) / 255) * 100
    clothing_bg_saturation_diff_pct = (abs(clothing_saturation - background_saturation) / 255) * 100
    
    # 점수 계산
    score = 100.0
    feedbacks = []
    
    # 채도 차이 평가
    if face_clothing_saturation_diff_pct < 5:
        score -= 20
        feedbacks.append("얼굴과 의상의 채도가 너무 비슷합니다. 의상의 채도를 조절해보세요")
    elif face_clothing_saturation_diff_pct > 15:
        score -= 10
        feedbacks.append("얼굴과 의상의 채도 차이가 너무 큽니다")
    
    if clothing_bg_saturation_diff_pct < 5:
        score -= 15
        feedbacks.append("의상과 배경의 채도가 너무 비슷합니다")
    elif clothing_bg_saturation_diff_pct > 15:
        score -= 10
        feedbacks.append("의상과 배경의 채도 차이가 너무 큽니다")
    
    print(f"색상 균형 점수: {score:.1f}")
    
    feedback = " / ".join(feedbacks) if feedbacks else None
    if feedback:
        print(f"색상 균형 피드백: {feedback}")
    else:
        print("색상 균형 피드백: 전체적인 색상 균형이 좋습니다")
    
    return score, feedback

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
        'height': 0.15,     # 전신 비율
        'composition': 0.2,  # 구도
        'vertical': 0.15,   # 수직 위치
        'focus': 0.15,      # 아웃포커싱
        'magnification': 0.15,  # 배율
        'exposure': 0.1        # 노출
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
    
    # 측정값들을 문자열로 변환
    more_info = "\n".join([f"{k}: {v}" for k, v in measurements.items()])
    
    # 최종 결과 형식 구성
    final_result = {
        "totalScore": round(total_score, 1),
        "feedback": feedbacks,
        "moreInfo": more_info
    }
    
    return final_result

import requests

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
        calculate_magnification_score(image, (body_x1, body_y1, body_x2, body_y2)),
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

            p = Path(p)
            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')
            s += '%gx%g ' % im.shape[2:]
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    if int(cls) == 0:  # person class
                        # Face detection
                        gray = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
                        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                        
                        if len(faces) > 0:
                            # 첫 번째 감지된 얼굴에 대해 평가 수행
                            result = process_single_detection(im0, xyxy, faces[0])
                            
                            if view_img:  # 시각화가 필요한 경우
                                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                                # 인물 박스 그리기
                                c = int(cls)
                                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                                annotator.box_label(xyxy, label, color=colors(c, True))
                                # 얼굴 박스 그리기
                                x, y, w, h = faces[0]
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