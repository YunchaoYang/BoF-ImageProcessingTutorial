'''
adapted from https://github.com/akash-agni/Real-Time-Object-Detection
author : akash-agni
'''
import torch
import numpy as np
import cv2
import pafy
from time import time


class ObjectDetection:
    """
    Class implements Yolo5 model to make inferences on a youtube video using Opencv2.
    """

    def __init__(self, url = None, webcamId = 0, out_file="Labeled_Video.mp4"):
        """
        Initializes the class with youtube url and output file.
        :param url: Has to be as youtube URL,on which prediction is made.
        :param out_file: A valid output file name.
        """
        self._URL = url
        self.model = self.load_model()
        self.classes = self.model.names
        self.out_file = out_file
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.webcamId = webcamId
        
        if url is not None:
            self.capture = self.get_video_from_url()
        elif webcamId is not None:
            self.capture = self.get_video_from_webcam()
            
    def get_video_from_url(self):
        """
        Creates a new video streaming object to extract video frame by frame to make prediction on.
        :return: opencv2 video capture object, with lowest quality frame available for video.
        """
        play = pafy.new(self._URL).streams[-1]
        assert play is not None
        return cv2.VideoCapture(play.url)
    
    def get_video_from_webcam(self):
        """
        Creates a new video streaming object to extract video frame by frame to make prediction on.
        :return: opencv2 video capture object, with lowest quality frame available for video.
        """
        return cv2.VideoCapture(self.webcamId)

    def load_model(self):
        """
        Loads Yolo5 model from pytorch hub.
        :return: Trained Pytorch model.
        """
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model

    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1].cpu().numpy(), results.xyxyn[0][:, :-1].cpu().numpy()
        return labels, cord

    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

        return frame

    def destroy(self):          
        self.cap.release()
        self.cap.destroyAllWindows()
        
    def __call__(self):
        """
        This function is called when class is executed, it runs the loop to read the video frame by frame,
        and write the output into a new file.
        :return: void
        """
        player = self.get_video_from_url()
        assert player.isOpened()
        x_shape = int(player.get(cv2.CAP_PROP_FRAME_WIDTH))
        y_shape = int(player.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        four_cc = cv2.VideoWriter_fourcc(*'avc1') # cv2.VideoWriter_fourcc(*'H264') # cv2.VideoWriter_fourcc(*'MP4V') # cv2.VideoWriter_fourcc(*"MJPG")  # cv2.VideoWriter_fourcc(*'VP90') 
        out = cv2.VideoWriter(self.out_file, four_cc, 20, (x_shape, y_shape))

        count = 0
        while True:
            start_time = time()
            ret, frame = player.read()
            
            # assert ret
            # check if frame is None
            if frame is None:
                #if True break the infinite loop
                break
            
            count += 1
            
            if count > 100:
                break
            
            results = self.score_frame(frame)
            frame = self.plot_boxes(results, frame)
            end_time = time()
            fps = 1/np.round(end_time - start_time, 3)
            print(f"Frame #{count}; speed: Frame Per Second : {fps}")
            out.write(frame)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        
        #out.release()

if __name__ == "__main__":
    # Create a new object and execute.
    a = ObjectDetection("https://www.youtube.com/watch?v=dwD1n7N7EAg")
    a()
