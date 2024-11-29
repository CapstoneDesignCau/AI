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
        'ratio': 0.25,      # 등신 비율
        'height': 0.2,      # 전신 비율
        'composition': 0.2,  # 구도
        'vertical': 0.2,    # 수직 위치
        'focus': 0.15       # 아웃포커싱
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
    
    # 측정값 저장
    measurements = {
        "전신 세로 길이": f"{body_height} pixels",
        "얼굴 세로 길이": f"{face_height} pixels",
        "등신 비율": f"{head_to_body_ratio:.2f} 등신",
        "전신/이미지 비율": f"{person_height_ratio:.1f}%"
    }
    
    # 각 평가 수행
    scores_and_feedbacks = [
        calculate_ratio_score(head_to_body_ratio),
        calculate_height_ratio_score(person_height_ratio),
        calculate_thirds_score(image_width, image_height, person_center_x),
        calculate_vertical_position_score(image_height, face_center_y, feet_y, head_y),
        analyze_focus_difference(image, (body_x1, body_y1, body_x2, body_y2), (x, y, w, h))
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