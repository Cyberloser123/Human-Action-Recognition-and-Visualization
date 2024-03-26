# Human-Action-Recognition-and-Visualization

Video Pose Extraction and 3D Skeleton Reconstruction is an innovative project that leverages state-of-the-art techniques in computer vision and 3D reconstruction to analyze human movements from video data. By combining the power of [MMpose](https://mmpose.readthedocs.io/en/latest/) for pose estimation and [Open3D](https://www.open3d.org/) for 3D visualization, this project offers a solution for understanding and visualizing human poses in three-dimensional space.

<center class="half">
<img src="https://i.imgur.com/VhMVNfb.gif" width="200"/><img src="https://i.imgur.com/OUGw0G9.gif" width="200"/> </center>


## Features
- Video Pose Extraction: The project can process videos to extract human poses frame by frame.

- 2D Skeleton Analysis: Utilizing mmpose, it analyzes the 2D skeleton information extracted from the videos.

- 3D Skeleton Reconstruction: The 2D skeleton data is transformed into 3D skeletons for a more comprehensive representation.

- Jump height estimation: Utilizing the information of the pixels to calculate the human vertical movement in real world.

- Visualization with Open3D: Finally, the reconstructed 3D skeletons are visualized using Open3D for easy interpretation and analysis.


## Requirements

mmpose == 1.3.1
</br>
open3d == 0.18.0
</br>
python == 3.10.11

## Install
Install the python packages with the following command:
```
python3 -m pip install -r requirements.txt
```

## Run

```
-> python main.py -i "file path"
```

Example:
```
-.> python main.py -i ./videos/test.mp4
```

[Video source](https://www.youtube.com/watch?v=kDOGb9C5kp0&)
