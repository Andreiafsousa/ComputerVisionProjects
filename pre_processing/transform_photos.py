"""Transformer images."""
import cv2
import numpy as np

from pathlib import Path


class TransformImages():
    """Extract images from video with 1 fps."""

    def __init__(self, video_source_path: str):
        """Extract images from video with 1 fps.

        Parameters
        ----------
        video_source_path: str
            path where we have the video files.
        """
        self.video_source_path = video_source_path

    def __call__(self):
        """Extract images from video with 1 fps."""

        pathlist = Path(self.video_source_path).rglob('*')
        print(pathlist)
        for path in pathlist:
            # because path is object not string
            file_name = str(path)
            print(file_name)
            self._extract_images(file_name)

    def _extract_images(self, file_name=str):

        vidcap = cv2.VideoCapture(file_name)
        count = 0
        success = True
        while success:
            vidcap.set(cv2.CAP_PROP_POS_MSEC, (count*1000))
            success, image = vidcap.read()

            # Stop when last frame is identified
            image_last = cv2.imread("frame{}.png".format(count-1))
            if np.array_equal(image, image_last):
                break

            cv2.imwrite("frame%d.png" % count, image)  # save frame as PNG file
            print('{}.sec reading a new frame: {} '.format(count, success))
            count += 1


if __name__ == "__main__":

    video_source_path = "/Users/andreiapfsousa/projects_andreiapfsousa/ComputerVisionProjects/videos_pilar/deitada"
    t = TransformImages(video_source_path)
    t()
