"""Transformer images."""
import os
import time
from pathlib import Path
import cv2

# Notes: Its necessary to have a folder images inside the directory!!!!

class TransformImages():
    """Extract images from video."""

    def __init__(self, directory: str, interval: int):
        """Extract images from video.

        Parameters
        ----------
        directory: str
            path where are the folders with video files.
        """
        self.directory = directory
        self.interval = interval
        # Log time
        self.time_start = time.time()

        try:
            os.path.isdir(self.directory)
        except OSError:
            print("Can't find the directory")
            raise

    def __call__(self):
        """Extract images from video."""
        for f in Path(self.directory).iterdir():
            pathIn = str(f)
            pathlist = Path(pathIn).glob("*.mp4")
            c = 0
            for path in pathlist:
                video_count = 1 + c
                print("video_count:", video_count)
                file_name = str(path)
                self._extract_images(file_name, video_count)
                c = c+1

    def _extract_images(self, file_name=str, video_count=int):
        # Start capturing the feed
        cap = cv2.VideoCapture(file_name)
        # Find the number of frames
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        print("Number of frames: ", video_length)
        count = 0
        print("Converting video..\n")
        # Start converting the video
        while cap.isOpened():
            # Extract the frame
            ret, frame = cap.read()
            if not ret:
                continue
            # Write the results back to output location.
            cv2.imwrite(str(file_name.replace(file_name.split("/")[7], "")) + "images/%#05d.jpg" % (count+video_count), frame)
            count = count + self.interval
            # If there are no more frames left
            if (count > (video_length-1)):
                # Log the time again
                time_end = time.time()
                # Release the feed
                cap.release()
                # Print stats

                print("Done extracting frames.\n%d frames extracted" % count)
                print("It took %d seconds forconversion." % (time_end-self.time_start))
                break


if __name__ == "__main__":
    directory = "/Users/andreiapfsousa/projects_andreiapfsousa/ComputerVisionProjects/videos_pilar"
    interval = 20
    t = TransformImages(directory, interval)
    t()
