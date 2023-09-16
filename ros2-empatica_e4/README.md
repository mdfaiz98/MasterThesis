ros2-empatica_e4
<p align="center">
<img src="/media/img/ros2_biosensor_pkg.png" width="700" >
</p>
This repository contains a ROS2 node for interfacing with the Empatica E4 wearable sensor. The node is designed to collect and publish various biosignals provided by the Empatica E4 device, facilitating the development of real-world Human-Robot Interaction (HRI) systems. The node standardizes biosensor HRI integration, lowers the technical barrier of entry, and expands the biosensor ecosystem in the robotics field.

Requirements
ROS2 foxy
Install Empatica E4 streaming server on an additional Window machine (Win10)
Install python libraries:
bash
Copy code
$ pip3 install open-e4-client pexpect websocket-client
Installation
bash
Copy code
$ cd ~
$ source /opt/ros/foxy/setup.bash
$ git clone 
$ cd ros2-empatica_e4
$ colcon build --symlink-install
$ source install/setup.bash
Usage
After building the workspace, source it and run the empatica_e4 node:

bash
Copy code
source install/setup.bash
ros2 run empatica_e4 empatica_e4_node
Contributors

