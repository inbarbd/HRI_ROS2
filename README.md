## HRI ROS2 Project

This packge tested on ros_eloquent Ubuntu 18.04.

The packge publish the pointer model to ROS and convert it to the camera frame of the user.
Copy the model to PointerModel.




Instractions:
1. In the launch flie set RealCam to False - if you work with your camera set it to True.
2. If RealCam if False change PKG_LOCATION to your computer path.
3. In  PublishModleRotetionData.py file set the camera frame name argumant. 
4. In RealTime_6_ros.py file change the path location.
5. In In RealTime_6_ros.py file change CAMERA_TOPIC to your camera RGB frame. This topic subscribe to the camera video and send the frame to the model.
4. launch launch_pointer_node.launch.py



To run pointer model in call StartAnimetionSrv: ros2 service call /StartAnimetionSrv std_srvs/srv/Empty\
to see the arrows in RVIz add Markr to visual.\

To see plot of the TF convertion from the model call PlotConvertionSrv: ros2 service call /PlotConvertionSrv std_srvs/srv/Empty

![demo](https://user-images.githubusercontent.com/57666167/209841750-a7796496-f718-472b-b1c1-6d4a393471c3.gif)