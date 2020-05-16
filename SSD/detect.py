import keyboard as keyboard
import numpy
import cv2
import torch.backends.cudnn as cudnn
from PIL import Image, ImageDraw, ImageFont
from SSD.utils import *
from torchvision import transforms


def detect(original_image, min_score, max_overlap, top_k, model, suppress=None):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.
    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """

    resize = transforms.Resize((300, 300))
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # Transform
    image = normalize(to_tensor(resize(original_image)))

    # Move to default device
    image = image.to(device)

    # Forward prop.
    predicted_locs, predicted_scores = model(image.unsqueeze(0))

    # Detect objects in SSD output
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k)

    # Move detections to the CPU
    det_boxes = det_boxes[0].to('cpu')

    # Transform to original image dimensions
    original_dims = torch.FloatTensor([original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    # Decode class integer labels
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

    # Detected objects dictionary
    det_objects = []

    # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
    if det_labels == ['background']:
        # Just return original image
        return original_image, det_objects

    # Annotate
    annotated_image = original_image
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.truetype("./calibril.ttf", 15)

    # Suppress specific classes, if needed
    for i in range(det_boxes.size(0)):
        if suppress is not None:
            if det_labels[i] in suppress:
                continue

        # Boxes
        box_location = det_boxes[i].tolist()
        draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
        draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[
            det_labels[i]])  # a second rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 2. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a third rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 3. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a fourth rectangle at an offset of 1 pixel to increase line thickness

        # Fill detected objects dictionary
        det_objects.append({"label": det_labels[i], "location": box_location})

        # Text
        text_size = font.getsize(det_labels[i].upper())
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                            box_location[1]]
        draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
        draw.text(xy=text_location, text=det_labels[i].upper(), fill='white',
                  font=font)
    del draw

    return annotated_image, det_objects


def infere(frame, object_counter, last_detected_objects, model):
    # Interfere with model
    cv2_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2_image = numpy.rot90(cv2_image)
    pil_image = Image.fromarray(cv2_image)
    pil_image, det_objects = detect(pil_image, min_score=0.7, max_overlap=0.45, top_k=200, model=model)
    cv2_image = numpy.array(pil_image)
    cv2_image = numpy.rot90(cv2_image, 3)
    cv2_image = cv2_image[:, :, ::-1].copy()

    # Count objects
    for det_object in det_objects:
        found = False
        for last_detected_object in last_detected_objects:
            if det_object["label"] == last_detected_object["label"]:
                if abs(sum(det_object["location"]) - sum(last_detected_object["location"])) <= 100:
                    found = True
                    break

        if not found:
            object_counter[det_object["label"]] += 1

    return cv2_image, det_objects, object_counter


if __name__ == '__main__':
    # Create Window
    cv2.namedWindow("SmartWarehouse")

    # Inference device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model checkpoint
    checkpoint = '../SSD/checkpoint/BEST_checkpoint_ssd300.pth.tar'
    checkpoint = torch.load(checkpoint)  # Use map_location=torch.device('cpu') as 2nd parameter on laptop
    model = checkpoint['model']
    model = model.to(device)
    model.eval()
    cudnn.benchmark = True
    cudnn.enabled = True

    vc = cv2.VideoCapture("C:\\Users\\Felix\\OneDrive\\Desktop\\Down\\b9.jpeg")

    _, image = vc.read()

    # Interfere with model
    cv2_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2_image = numpy.rot90(cv2_image)
    pil_image = Image.fromarray(cv2_image)
    pil_image, det_objects = detect(pil_image, min_score=0.75, max_overlap=0.5, top_k=1000, model=model)
    cv2_image = numpy.array(pil_image)
    cv2_image = numpy.rot90(cv2_image, 3)
    cv2_image = cv2_image[:, :, ::-1].copy()

    # Display frame
    cv2.imshow("SmartWarehouse", cv2_image)
    cv2.waitKey(0)
