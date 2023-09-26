#!/usr/bin/env python2.7

import rospy
from std_msgs.msg import Float32, Float32MultiArray
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage
from collections import OrderedDict  # Import OrderedDict for ordered dictionary

class DataCollector:
    def __init__(self):
        rospy.init_node("data_collector_node", anonymous=True)

        # Define the topics and their corresponding message types
        topics = OrderedDict([
            ("/biosensors/empatica_e4/bvp", Float32),
            ("/biosensors/empatica_e4/st", Float32),
            ("/biosensors/empatica_e4/gsr", Float32),
            ("/biosensors/empatica_e4/hr", Float32),
            ("/biosensors/empatica_e4/ibi", Float32),
            ("/biosensors/empatica_e4/acc", Float32MultiArray),
            ("/tf", TFMessage)  # Include /tf in the list of topics
        ])

        # Store the latest data for different topics
        self.latest_data = {topic: None for topic in topics}

        # Create a dictionary to store data sets for each timestamp
        self.data_sets = {}

        # Create a flag to indicate if BVP data has been received
        self.bvp_data_received = False

        # Create subscribers for all topics
        self.subscribers = {}
        for topic, msg_type in topics.items():
            self.subscribers[topic] = rospy.Subscriber(topic, msg_type, self.custom_callback, callback_args=topic)

    def custom_callback(self, data, topic):
        # Callback function for all topics
        rospy.loginfo("Received %s data: %s", topic, data)
        self.latest_data[topic] = data

        # If BVP data is received, create a data set for the current timestamp
        if topic == "/biosensors/empatica_e4/bvp":
            current_time = rospy.Time.now()
            data_set = {"timestamp": current_time}
            for topic, latest_data in self.latest_data.items():
                if latest_data is not None:
                    data_set[topic] = latest_data
                else:
                    data_set[topic] = None
                    rospy.loginfo("No data received for %s yet.", topic)

            # Store the data set for this timestamp
            self.data_sets[current_time] = data_set

            # Log the complete data set
            rospy.loginfo("Data set: %s", data_set)

            # Set the flag to indicate that BVP data has been received
            self.bvp_data_received = True

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = DataCollector()
        node.run()
    except rospy.ROSInterruptException:
        pass
