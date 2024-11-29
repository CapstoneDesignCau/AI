import argparse
import os
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
    ideal_min = 7.0
    ideal_max = 10.0
    
    if ideal_min <= head_to_body_ratio <= ideal_max:
        comment = "적절한 등신 비율입니다."
    elif head_to_body_ratio < ideal_min:
        comment = "카메라 위치를 낮추어 촬영하면 다리가 더 길어 보일 수 있습니다."
    else:
        comment = "카메라 위치를 높여서 촬영하면 전신 비율이 더 자연스러워질 수 있습니다."
    
    if ideal_min <= head_to_body_ratio <= ideal_max:
        score = 100.0
    else:
        if head_to_body_ratio < ideal_min:
            difference = ideal_min - head_to_body_ratio
        else:
            difference = head_to_body_ratio - ideal_max
        penalty = min(difference * 20, 100)
        score = max(0, 100 - penalty)
    
    return score, comment

def calculate_vertical_position_score(image_height, face_center_y, feet_y, head_y):
    score = 100.0
    penalties = []
    
    image_center_y = image_height / 2
    face_center_distance = abs(face_center_y - image_center_y)
    face_center_ratio = face_center_distance / image_height
    
    feet_distance = abs(feet_y - image_height)
    feet_ratio = feet_distance / image_height
    
    face_penalty = face_center_ratio * 100
    feet_penalty = feet_ratio * 100
    
    total_penalty = min(face_penalty * 2, 50) + min(feet_penalty * 2, 50)
    score = max(0, 100 - total_penalty)
    
    if score >= 80:
        comment = "인물의 수직 위치가 적절합니다."
    elif feet_ratio > 0.1:
        comment = "발이 프레임 아래쪽에 더 가깝게 위치하도록 구도를 잡아보세요."
    else:
        comment = "얼굴이 화면 중앙에 오도록 구도를 조정해보세요."
    
    return score, comment

def calculate_thirds_score(image_width, image_height, person_center_x):
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
    
    if normalized_distance < 0.05:
        comment = "인물이 3등분선 위치에 잘 배치되어 있습니다."
    elif normalized_distance < 0.15:
        comment = "구도가 양호하나, 3등분선에 더 가깝게 위치하면 더 좋을 것 같습니다."
    else:
        comment = "인물을 화면의 1/3 지점이나 중앙에 위치시키면 더 안정적인 구도가 될 수 있습니다."
    
    return score, normalized_distance, comment

def calculate_height_ratio_score(person_height_ratio):
    ideal_min = 65.0
    ideal_max = 85.0
    
    if ideal_min <= person_height_ratio <= ideal_max:
        comment = "인물이 프레임에 잘 맞게 촬영되었습니다."
    elif person_height_ratio < ideal_min:
        comment = "인물을 좀 더 크게 촬영하면 좋겠습니다. 카메라를 더 가까이 가져가보세요."
    else:
        comment = "인물이 프레임에 비해 너무 큽니다. 카메라를 조금 더 멀리서 촬영해보세요."
    
    if ideal_min <= person_height_ratio <= ideal_max:
        score = 100.0
    else:
        if person_height_ratio < ideal_min:
            difference = ideal_min - person_height_ratio
        else:
            difference = person_height_ratio - ideal_max
        penalty = min(difference * 2, 100)
        score = max(0, 100 - penalty)
    
    return score, comment

def analyze_focus_difference(image, person_bbox, face_bbox):
    def get_blur_score(roi):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    body_x1, body_y1, body_x2, body_y2 = person_bbox
    face_x, face_y, face_w, face_h = face_bbox
    
    # 얼굴 영역의 선명도 측정
    face_roi = image[face_y:face_y+face_h, face_x:face_x+face_w]
    face_sharpness = get_blur_score(face_roi)
    
    # 배경 영역 마스크 생성 (인물 영역을 제외한 부분)
    mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
    mask[body_y1:body_y2, body_x1:body_x2] = 0
    
    # 배경 영역의 선명도 측정
    background = cv2.bitwise_and(image, image, mask=mask)
    background_sharpness = get_blur_score(background)
    
    # 선명도 차이 계산
    sharpness_diff = face_sharpness - background_sharpness
    
    # 점수 계산 (0-100 범위로 정규화)
    max_diff = 1000  # 이 값은 테스트를 통해 조정 필요
    score = min(100, max(0, (sharpness_diff / max_diff) * 100))
    
    # 상태 분석 및 코멘트 생성
    if score < 30:
        status = "no_focus"
        comment = "아웃포커싱 효과를 활용하면 인물을 더 돋보이게 할 수 있습니다."
    elif score > 80:
        status = "too_much_focus"
        comment = "배경의 아웃포커싱이 과도합니다. 조금 더 줄여보세요."
    else:
        status = "good_focus"
        comment = "아웃포커싱이 적절합니다."
    
    return {
        'score': score,
        'status': status,
        'comment': comment,
        'face_sharpness': face_sharpness,
        'background_sharpness': background_sharpness,
        'difference': sharpness_diff
    }

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
    vid_path, vid_writer = [None] * bs, [None] * bs
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
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            # Face detection
            gray = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Process detections
                for *xyxy, conf, cls in reversed(det):
                    if int(cls) == 0:  # if person class
                        # Calculate body height and positions
                        body_x1, body_y1, body_x2, body_y2 = [int(x) for x in xyxy]
                        body_height = body_y2 - body_y1
                        person_center_x = (body_x1 + body_x2) / 2
                
                        # Calculate person height ratio to image height
                        image_height = im0.shape[0]
                        image_width = im0.shape[1]
                        person_height_ratio = (body_height / image_height) * 100
                
                        # Calculate all scores
                        composition_score, distance, composition_comment = calculate_thirds_score(image_width, image_height, person_center_x)
                        height_ratio_score, height_comment = calculate_height_ratio_score(person_height_ratio)
                
                        # Process face detections
                        for (x, y, w, h) in faces:
                            face_height = h
                            head_to_body_ratio = body_height / face_height
                
                            # Calculate face center and feet position
                            face_center_y = y + (h / 2)
                            feet_y = body_y2
                            head_y = body_y1
                
                            # Calculate all scores
                            ratio_score, ratio_comment = calculate_ratio_score(head_to_body_ratio)
                            vertical_score, vertical_comment = calculate_vertical_position_score(image_height, face_center_y, feet_y, head_y)
                            focus_analysis = analyze_focus_difference(im0, (body_x1, body_y1, body_x2, body_y2), (x, y, w, h))
                
                            # Print results
                            print(f"\n전신 세로 길이: {body_height} pixels")
                            print(f"얼굴 세로 길이: {face_height} pixels")
                            print(f"등신 비율: {head_to_body_ratio:.2f} 등신")
                            print(f"등신 점수: {ratio_score:.1f}")
                            print(f"등신 코멘트: {ratio_comment}")
                            print(f"전신/이미지 비율: {person_height_ratio:.1f}%")
                            print(f"전신 비율 점수: {height_ratio_score:.1f}")
                            print(f"전신 비율 코멘트: {height_comment}")
                            print(f"구도 점수: {composition_score:.1f}")
                            print(f"구도 코멘트: {composition_comment}")
                            print(f"수직 위치 점수: {vertical_score:.1f}")
                            print(f"수직 위치 코멘트: {vertical_comment}")
                            print(f"아웃포커싱 점수: {focus_analysis['score']:.1f}")
                            print(f"아웃포커싱 코멘트: {focus_analysis['comment']}")
                
                            # Draw face box
                            cv2.rectangle(im0, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
                            # Calculate font scale based on image size
                            font_scale = min(image_width, image_height) * 0.001
                            thickness = max(1, int(min(image_width, image_height) * 0.004))
                
                            # Add ratio text with main scores
                            ratio_text = (f"Ratio: {head_to_body_ratio:.2f}({ratio_score:.1f}) | "
                                        f"Height: {person_height_ratio:.1f}%({height_ratio_score:.1f}) | "
                                        f"Comp: {composition_score:.1f} | Vert: {vertical_score:.1f}")
                
                            # 화면 최상단 중앙에 메인 텍스트 배치
                            text_size = cv2.getTextSize(ratio_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                            text_x = im0.shape[1]//2 - text_size[0]//2
                            text_y = text_size[1] + 10
                
                            # Add text to image
                            cv2.putText(im0, ratio_text,
                                      (text_x, text_y),
                                      cv2.FONT_HERSHEY_SIMPLEX,
                                      font_scale,
                                      (0, 0, 0),
                                      thickness)
                
                            # Add focus comment
                            comment_scale = font_scale * 0.8
                            comment_text = f"Focus: {focus_analysis['comment']}"
                            comment_size = cv2.getTextSize(comment_text, cv2.FONT_HERSHEY_SIMPLEX, comment_scale, thickness)[0]
                            comment_x = im0.shape[1]//2 - comment_size[0]//2
                            comment_y = im0.shape[0] - 20
                
                            cv2.putText(im0, comment_text,
                                      (comment_x, comment_y),
                                      cv2.FONT_HERSHEY_SIMPLEX,
                                      comment_scale,
                                      (0, 0, 255),
                                      thickness)

                            # Draw center line
                            cv2.line(im0, (0, int(image_height/2)), (image_width, int(image_height/2)), 
                                   (128, 128, 128), 1)

                        if save_txt:
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)
                            with open(f'{txt_path}.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or save_crop or view_img:
                            c = int(cls)
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(10000)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)

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