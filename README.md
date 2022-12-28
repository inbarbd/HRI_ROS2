## HRI ROS2 Project

This packge tested on ros_eloquent Ubuntu 18.04.\

The packge publish the pointer model to ROS and convert it to the camera frame of the user.
Copy the model to PointerModel.

Instractions:
1. In the launch flie set RealCam to bool value. defult is False and PKG_LOCATION parameter.
2. In  PublishModleRotetionData.py file set the camera frame name argumant.
3. In Ros2_Eden_complete_model.py file change the path location.
4. launch launch_pointer_node.launch.py

To run pointer model in call StartAnimetionSrv: ros2 service call /StartAnimetionSrv std_srvs/srv/Empty\
To see plot of the TF convertion from the model call PlotConvertionSrv: ros2 service call /PlotConvertionSrv std_srvs/srv/Empty\

