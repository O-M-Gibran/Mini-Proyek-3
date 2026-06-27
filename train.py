import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

def main():
    print("Current Working Directory:", os.getcwd())
    model = YOLO('yolov8n.pt')

    print("Starting training")
    results = model.train(
        data='data.yaml',
        epochs=100,
        patience=15,
        imgsz=640,
        project='./runs',
        name='atm_training',
        workers=0,
    )

    print("Training Complete!")
    model.val()

    test_image_path = os.path.join(
        'dataset',
        'valid',
        'images',
        'dc187924523d4b8fad9a148e9a4f431e_jpg.rf.23d9a3e2bf94ee91a4515540077412ac.jpg',
    )

    if os.path.exists(test_image_path):
        predictions = model.predict(source=test_image_path, conf=0.25)

        for result in predictions:
            img = result.plot()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            plt.imshow(img)
            plt.axis('off')
            plt.show()

    model.export(format='onnx')


if __name__ == '__main__':
    main()