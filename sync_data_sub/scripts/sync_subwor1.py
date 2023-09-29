#!/usr/bin/env python2.7

import rospy
from e4_msgs.msg import Float32WithHeader, Float32MultiArrayWithHeader
from tf2_msgs.msg import TFMessage
from collections import OrderedDict  

class DataCollector:
    def __init__(self):
        rospy.init_node("data_collector_node", anonymous=True)

        topics = OrderedDict([
            ("/biosensors/empatica_e4/bvp", Float32WithHeader),
            ("/biosensors/empatica_e4/st", Float32WithHeader),
            ("/biosensors/empatica_e4/gsr", Float32WithHeader),
            ("/biosensors/empatica_e4/hr", Float32WithHeader),
            ("/biosensors/empatica_e4/ibi", Float32WithHeader),
            ("/biosensors/empatica_e4/acc", Float32MultiArrayWithHeader),
            ("/tf", TFMessage)
        ])

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
        # Extract timestamp from the header
        if topic == "/tf":
            topic_timestamp = data.transforms[0].header.stamp
        else:
            topic_timestamp = data.header.stamp

        # If within the 5-second waiting period, just update the data
        if self.waiting_period and (topic_timestamp - self.start_time).to_sec() < 5:
            self.latest_data[topic] = data
            self.timestamps[topic] = topic_timestamp
            return

        # If just ending the waiting period, update the flag
        if self.waiting_period:
            self.waiting_period = False

        # If the topic is /tf, process the transformations and add to the data set
        if topic == "/tf":
            tf_data = []
            for transform_stamped in data.transforms:
                child_frame_id = transform_stamped.child_frame_id
                transform = transform_stamped.transform
                # Store the transformation data for the child frame
                tf_data.append({
                    "header": transform_stamped.header,
                    "child_frame_id": child_frame_id,
                    "translation": transform.translation,
                    "rotation": transform.rotation
                })
            
            # Replace the TF data in the latest data for /tf
            self.latest_data["/tf"] = tf_data

        # Update the latest data
        self.latest_data[topic] = data
        self.timestamps[topic] = topic_timestamp

        # If the data is from BVP topic, log the dataset
        if topic == "/biosensors/empatica_e4/bvp":
            data_set = {"timestamp": topic_timestamp, "data_timestamps": self.timestamps}
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
