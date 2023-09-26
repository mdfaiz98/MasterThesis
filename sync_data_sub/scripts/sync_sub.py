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
            ("/tf", TFMessage)
        ])

        # Store the latest data for different topics with initial value as "None Received"
        self.latest_data = {topic: "None Received" for topic in topics}
        self.timestamps = {topic: None for topic in topics}  # Timestamps for each topic data

        # Create subscribers for all topics
        self.subscribers = {}
        for topic, msg_type in topics.items():
            self.subscribers[topic] = rospy.Subscriber(topic, msg_type, self.custom_callback, callback_args=topic)

        # Initialize the start time and the waiting flag
        self.start_time = rospy.Time.now()
        self.waiting_period = True

    def custom_callback(self, data, topic):
        # If within the 5-second waiting period, just update the data
        if self.waiting_period and (rospy.Time.now() - self.start_time).to_sec() < 5:
            self.latest_data[topic] = data
            return
        
        # If just ending the waiting period, update the flag
        if self.waiting_period:
            self.waiting_period = False

        # Update the latest data and timestamp for the topic
        self.latest_data[topic] = data
        self.timestamps[topic] = rospy.Time.now()

        # If the data is from BVP topic, log the dataset
        if topic == "/biosensors/empatica_e4/bvp":
            # Create the dataset
            current_time = rospy.Time.now()
            data_set = {"timestamp": current_time, "data_timestamps": self.timestamps}
            data_set.update(self.latest_data)
            
            # Format and log the dataset
            formatted_data = "\n".join(["{}: {}".format(key, value) for key, value in data_set.items()])
            rospy.loginfo("Data set:\n%s", formatted_data)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = DataCollector()
        node.run()
    except rospy.ROSInterruptException:
        pass
