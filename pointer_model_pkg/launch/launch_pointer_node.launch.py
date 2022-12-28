#!/usr/bin/env /home/inbarm/dev_ws/src/HRI_ROS2/pointer_model_pkg/hri_ros2/bin/python3

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import LaunchConfiguration
import launch
import os
import numpy as np


''' Enter Your packge location '''
PKG_LOCATION = '/home/inbarm/dev_ws/src/HRI_ROS2/pointer_model_pkg/'
RealCam = False


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    
    ld = LaunchDescription()
    rviz_node = Node(
            package='rviz2',
            # namespace='',
            node_executable='rviz2',
            name='rviz2',
            arguments=['-d' + os.path.join(get_package_share_directory('pointer_model_pkg'), 'config', 'pointer_model_pkg_rviz.rviz')]
        )
    if not RealCam:
        urdf = PKG_LOCATION+'urdf/camera_link.xacro'
        with open(urdf, 'r') as infp:
            robot_desc = infp.read()
        static_camera_tf_node = Node(package = "tf2_ros", 
                        node_executable = "static_transform_publisher",
                        arguments = ["0", "0", "1", "0", "0", str(np.pi/2), "map", "camera_frame_motive"])
        static_camera_tf_node1 = Node(package = "tf2_ros", 
                        node_executable = "static_transform_publisher",
                        arguments = ["0", "0", "0", "0", str(np.pi), str(-np.pi/2), "camera_frame_motive", "camera_frame"])
        ld.add_action(static_camera_tf_node)
        ld.add_action(static_camera_tf_node1)
    
    pointer_model_node_node = Node(
            package='pointer_model_pkg',
            node_executable='pointer_model_node',
            name='pointer_model_node_node',
            output='screen')
    pub_model_to_robot_node = Node(
            package='pointer_model_pkg',
            node_executable='pub_model_to_robot',
            name='pub_model_to_robot_node',
            output='screen')


    ld.add_action(pointer_model_node_node)
    ld.add_action(pub_model_to_robot_node)
    ld.add_action(rviz_node)
    
    return ld