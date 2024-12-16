# !nvidia-smi
# %pip install ultralytics supervision roboflow inference
#
# from google.colab import userdata
# from roboflow import Roboflow
# from IPython.display import Image as IPyImage
# from IPython.display import Image as  display
# import glob
# import random
# import cv2
# import supervision as sv
# import IPython
# import inference
# import os
# import ultralytics
#
# HOME = '/content'
# print(HOME)
#
# ultralytics.checks()
#
# rf = Roboflow(api_key=userdata.get('RoboFlowKey'))
# project = rf.workspace("capstone-roipa").project("frc-automatic-scouting")
# version = project.version(13)
# dataset = version.download("yolov11")
#
# %cd {HOME}
#
# !yolo task=detect mode=train model=yolo11s.pt data={dataset.location}/data.yaml epochs=300 imgsz=640 plots=True save=True amp=True
#
# !ls {HOME}/runs/detect/train/
#
#
# IPyImage(filename=f'{HOME}/runs/detect/train/confusion_matrix.png', width=600)
# IPyImage(filename=f'{HOME}/runs/detect/train/results.png', width=600)
# IPyImage(filename=f'{HOME}/runs/detect/train/val_batch0_pred.jpg', width=600)
#
# !yolo task=detect mode=val model={HOME}/runs/detect/train/weights/best.pt data={dataset.location}/data.yaml
#
# !yolo task=detect mode=predict model={HOME}/runs/detect/train/weights/best.pt conf=0.25 source={dataset.location}/test/images save=True
#
#
# latest_folder = max(glob.glob('/content/runs/detect/predict*/'), key=os.path.getmtime)
# for img in glob.glob(f'{latest_folder}/*.jpg')[:3]:
#     display(IPyImage(filename=img, width=600))
#     print("\n")
#
# project.version(dataset.version).deploy(model_type="yolov11", model_path=f"{HOME}/runs/detect/train/")
#
# model_id = project.id.split("/")[1] + "/" + dataset.version
# model = inference.get_model(model_id, userdata.get('RoboFlowKey'))
#
# # Location of test set images
# test_set_loc = dataset.location + "/test/images/"
# test_images = os.listdir(test_set_loc)
#
# # Run inference on 4 random test images, or fewer if fewer images are available
# for img_name in random.sample(test_images, min(4, len(test_images))):
#     print("Running inference on " + img_name)
#
#     # Load image
#     image = cv2.imread(os.path.join(test_set_loc, img_name))
#
#     # Perform inference
#     results = model.infer(image, confidence=0.4, overlap=50)[0]
#     detections = sv.Detections.from_inference(results)
#
#     # Annotate boxes and labels
#     box_annotator = sv.BoxAnnotator()
#     label_annotator = sv.LabelAnnotator()
#     annotated_image = box_annotator.annotate(scene=image, detections=detections)
#     annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
#
#     # Display annotated image
#     _, ret = cv2.imencode('.jpg', annotated_image)
#     i = IPython.display.Image(data=ret)
#     IPython.display.display(i)