#!/usr/bin/env python2.7

import rospy
from std_msgs.msg import Float32, Float32MultiArray
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage

class DataCollector:
    def __init__(self):
        rospy.init_node("data_collector_node", anonymous=True)

        # Define the topics and their corresponding message types
        topics = {
            "/biosensors/empatica_e4/bvp": Float32,
            "/biosensors/empatica_e4/gsr": Float32,
            "/biosensors/empatica_e4/hr": Float32,
            "/biosensors/empatica_e4/st": Float32,
            "/biosensors/empatica_e4/acc": Float32MultiArray,
            "/tf": TFMessage
        }

        # Store the latest data for different topics
        self.latest_data = {topic: None for topic in topics}

        # Create subscribers for all topics
        self.subscribers = {}
        for topic, msg_type in topics.items():
            self.subscribers[topic] = rospy.Subscriber(topic, msg_type, self.custom_callback, callback_args=topic)

        # Create a timer to log /tf data at 60 Hz
        self.tf_timer = rospy.Timer(rospy.Duration(1.0 / 60), self.log_tf_data)

    def custom_callback(self, data, topic):
        # Callback function for all topics except /biosensors/empatica_e4/bvp and /tf
        rospy.loginfo("Received %s data: %s", topic, data)
        self.latest_data[topic] = data

    def bvp_callback(self, data):
        # Callback function for the /biosensors/empatica_e4/bvp topic
        current_time = rospy.Time.now()
        rospy.loginfo("Received BVP data at time %s: %s", current_time, data.data)

        # Log the latest data for other topics
        for topic, latest_data in self.latest_data.items():
            if latest_data is not None:
                rospy.loginfo("Latest %s data at time %s: %s", topic, current_time, latest_data)
            else:
                rospy.loginfo("No data received for %s yet.", topic)

    def log_tf_data(self, event):
        # Callback function for logging /tf data at 60 Hz
        # You can downsample the /tf data here and log it at the desired rate
        # For simplicity, let's assume you just log the latest /tf data
        tf_data = self.latest_data["/tf"]
        if tf_data is not None:
            current_time = rospy.Time.now()
            rospy.loginfo("Latest TF data at time %s: %s", current_time, tf_data)
        else:
            rospy.loginfo("No /tf data received yet.")

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = DataCollector()
        node.run()
    except rospy.ROSInterruptException:
        pass
