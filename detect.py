import argparse
import os
import sys
from pathlib import Path
import cv2
import torch
from ultralytics.utils.plotting import Annotator, colors, save_one_box
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements,
                         colorstr, increment_path, non_max_suppression, print_args, scale_boxes,
                         strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, smart_inference_mode

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

@smart_inference_mode()
def run(
        weights=ROOT / "yolov5s.pt",
        source=ROOT / "data/images",
        data=ROOT / "data/coco128.yaml",
        imgsz=(640, 640),
        conf_thres=0.25,
        iou_thres=0.45,
        max_det=1000,
        device="",
        view_img=False,
        save_txt=False,
        nosave=False,
        classes=[0],  # COCO 데이터셋에서 사람은 클래스 0
        agnostic_nms=False,
        augment=False,
        visualize=False,
        update=False,
        project=ROOT / "runs/detect",
        name="exp",
        exist_ok=False,
        line_thickness=3,
        hide_labels=False,
        hide_conf=False,
        half=False,
        dnn=False,
        vid_stride=1,
):
    source = str(source)
    save_img = not nosave and not source.endswith(".txt")
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)

    # 디렉토리 설정
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    # 모델 로드
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    # Haar Cascade 얼굴 감지기 로드
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

    # 추론 실행
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
    
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(device)
            im = im.half() if model.fp16 else im.float()
            im /= 255
            if len(im.shape) == 3:
                im = im[None]

        # 추론
        with dt[1]:
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # 결과 처리
        for i, det in enumerate(pred):
            seen += 1
            if webcam:
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)
            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")
            s += "%gx%g " % im.shape[2:]
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            # 얼굴 감지
            gray = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                cv2.rectangle(im0, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 파란색으로 얼굴 표시

            if len(det):
                # 박스 크기 조정
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # 결과 출력
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # 텍스트 파일에 결과 저장
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # 이미지에 박스 추가
                        c = int(cls)
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))

            # 결과 이미지 스트림
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(10000)

            # 결과 저장
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path[i] != save_path:
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()
                        if vid_cap:
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / 'yolov5s.pt', help='모델 가중치 경로')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='파일/디렉토리/URL/웹캠 소스')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='데이터셋.yaml 경로')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='추론 크기')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='신뢰도 임계값')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU 임계값')
    parser.add_argument('--max-det', type=int, default=1000, help='이미지당 최대 감지 수')
    parser.add_argument('--device', default='', help='cuda 장치')
    parser.add_argument('--view-img', action='store_true', help='결과 보기')
    parser.add_argument('--save-txt', action='store_true', help='결과를 *.txt로 저장')
    parser.add_argument('--nosave', action='store_true', help='이미지/비디오 저장하지 않음')
    parser.add_argument('--classes', default=[0], type=int, help='사람 클래스만 필터링')
    parser.add_argument('--agnostic-nms', action='store_true', help='클래스-무관 NMS')
    parser.add_argument('--augment', action='store_true', help='증강된 추론')
    parser.add_argument('--visualize', action='store_true', help='특징 시각화')
    parser.add_argument('--update', action='store_true', help='모든 모델 업데이트')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='결과 저장 경로')
    parser.add_argument('--name', default='exp', help='결과 저장 폴더명')
    parser.add_argument('--exist-ok', action='store_true', help='기존 폴더 덮어쓰기')
    parser.add_argument('--line-thickness', default=3, type=int, help='경계 상자 두께')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='라벨 숨기기')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='신뢰도 숨기기')
    parser.add_argument('--half', action='store_true', help='FP16 half-precision 추론')
    parser.add_argument('--dnn', action='store_true', help='ONNX 추론을 위한 OpenCV DNN 사용')
    parser.add_argument('--vid-stride', type=int, default=1, help='비디오 프레임 간격')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    print_args(vars(opt))
    return opt

def main(opt):
    ##check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)