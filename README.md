# People Counting with Surveillance Camera System
This project was made with YOLOv8 and ByteTrack Algorithms.

Tools and libraries used: Python, OpenCV, NumPy, Pandas, Matplotlib, Supervision Library

# Summary
<p align="justify">This project develops an intelligent people counting system capable of detecting, tracking, and counting individuals in surveillance videos. The system leverages YOLOv8 for high-accuracy human detection and ByteTrack for multi-object tracking, ensuring that each person is assigned a unique identity throughout the video sequence.</p>

<p align="justify">By integrating object detection with tracking algorithms, the system can monitor pedestrian movement and calculate the number of people entering or leaving a designated area. Performance evaluation shows that the proposed approach provides accurate counting results while maintaining real-time processing capabilities. The project demonstrates the effectiveness of modern computer vision techniques in automated surveillance and crowd analytics applications.</p>

# About this project
<p align="justify">This project implements a real-time people counting system using surveillance camera footage by combining YOLOv8 for object detection and ByteTrack for multi-object tracking. The goal is to automatically detect, track, and count individuals as they move through a monitored area, providing an efficient solution for crowd monitoring, occupancy analysis, and smart surveillance applications.</p>

<p align="justify">The system first utilizes YOLOv8 to accurately detect people in each video frame. Detected individuals are then assigned unique tracking IDs using the ByteTrack algorithm, allowing the system to maintain consistent identities across consecutive frames. By tracking the movement of each person and monitoring predefined counting zones or virtual lines, the system can accurately count entries and exits while avoiding duplicate counts.</p>

<p align="justify">The project includes video preprocessing, real-time object detection, multi-object tracking, and counting logic implementation. Experimental results demonstrate that the combination of YOLOv8 and ByteTrack achieves reliable detection and tracking performance even in crowded environments, enabling accurate people counting and movement analysis. The system can be applied to various real-world scenarios, including retail analytics, building occupancy management, public transportation monitoring, and smart city surveillance.</p>
