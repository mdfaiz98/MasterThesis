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
            "/tf":  TFMessage
        }

        # store the latest data for different topics
        self.latest_data = {topic: None for topic in topics}

        # Create subscribers for all topics
        self.subscribers = {}
        for topic, msg_type in topics.items():
            self.subscribers[topic] = rospy.Subscriber(topic, msg_type, self.custom_callback, callback_args=topic)

    def custom_callback(self, data, topic):
        # Callback function for all topics except /biosensors/empatica_e4/bvp and /tf
        rospy.loginfo("Received %s data: %s", topic, data)
        self.latest_data[topic] = data

    def bvp_callback(self, data):
        # Callback function for the /biosensors/empatica_e4/bvp topic
        current_time = rospy.Time.now()
        #rospy.loginfo("Received BVP data: %s", data.data)
        rospy.loginfo("Received BVP data at time %s: %s", current_time, data.data)


        # Log the latest data for other topics
        for topic, latest_data in self.latest_data.items():
            if latest_data is not None:
                rospy.loginfo("Latest %s data at time %s: %s", current_time, topic, latest_data)
            else:
                rospy.loginfo("No data received for %s yet.", topic)

    def tf_callback(self, data):
         pass

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = DataCollector()
        node.run()
    except rospy.ROSInterruptException:
        pass
