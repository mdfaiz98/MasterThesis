#!/usr/bin/env python2.7

import rospy
from e4_msgs.msg import Float32WithHeader, Float32MultiArrayWithHeader
from tf2_msgs.msg import TFMessage
from e4_msgs.msg import ChildFramePos, AggregatedData
from collections import OrderedDict

class DataCollector:
    def __init__(self):
        rospy.init_node("data_collector_node", anonymous=True)

        self.pub = rospy.Publisher("/aggregated_data", AggregatedData, queue_size=10)  # New Publisher

        topics = OrderedDict([
            ("/biosensors/empatica_e4/bvp", Float32WithHeader),
            ("/biosensors/empatica_e4/st", Float32WithHeader),
            ("/biosensors/empatica_e4/gsr", Float32WithHeader),
            ("/biosensors/empatica_e4/hr", Float32WithHeader),
            ("/biosensors/empatica_e4/ibi", Float32WithHeader),
            ("/biosensors/empatica_e4/acc", Float32MultiArrayWithHeader),
            ("/tf", TFMessage)
        ])

        self.latest_data = {topic: float('nan') for topic in topics}
        self.latest_data["/tf"] = []
        self.timestamps = {topic: None for topic in topics}

        self.subscribers = {}
        for topic, msg_type in topics.items():
            self.subscribers[topic] = rospy.Subscriber(topic, msg_type, self.custom_callback, callback_args=topic)

        #self.start_time = rospy.Time.now()
        #self.waiting_period = True
        self.all_child_frames = set()

    def custom_callback(self, data, topic):
        if topic == "/tf":
            topic_timestamp = data.transforms[0].header.stamp
            
            for transform_stamped in data.transforms:
                child_frame_id = transform_stamped.child_frame_id
                self.all_child_frames.add(child_frame_id)

                transform = transform_stamped.transform
                existing_data = [item for item in self.latest_data["/tf"] if item["child_frame_id"] == child_frame_id]
                
                if existing_data:
                    existing_data[0].update({
                        "header": transform_stamped.header,
                        "translation": transform.translation,
                        "rotation": transform.rotation
                    })
                else:
                    self.latest_data["/tf"].append({
                        "header": transform_stamped.header,
                        "child_frame_id": child_frame_id,
                        "translation": transform.translation,
                        "rotation": transform.rotation
                    })
        else:
            topic_timestamp = data.header.stamp
            self.latest_data[topic] = data.data
            
            

        self.timestamps[topic] = topic_timestamp

        # if topic == "/biosensors/empatica_e4/bvp":
        #     data_set = {
        #         "timestamp": topic_timestamp,
        #         "data_timestamps": self.timestamps,
        #         "all_child_frames": list(self.all_child_frames)
        #     }
        #     data_set.update(self.latest_data)
        #     formatted_data = "\n".join(["{}: {}".format(key, value) for key, value in data_set.items()])
        #     rospy.loginfo("Data set:\n%s", formatted_data)

        if topic == "/biosensors/empatica_e4/bvp":
            aggregated_data_msg = AggregatedData()
            aggregated_data_msg.header.stamp = topic_timestamp
            aggregated_data_msg.bvp = self.latest_data["/biosensors/empatica_e4/bvp"]
            aggregated_data_msg.st = self.latest_data["/biosensors/empatica_e4/st"]
            aggregated_data_msg.gsr = self.latest_data["/biosensors/empatica_e4/gsr"]
            aggregated_data_msg.hr = self.latest_data["/biosensors/empatica_e4/hr"]
            aggregated_data_msg.ibi = self.latest_data["/biosensors/empatica_e4/ibi"]
            aggregated_data_msg.acc = self.latest_data["/biosensors/empatica_e4/acc"]
            for tf_data in self.latest_data["/tf"]:
                pose_msg = ChildFramePos()
                pose_msg.header = tf_data["header"]
                pose_msg.child_frame_id = tf_data["child_frame_id"]
                pose_msg.transform.translation = tf_data["translation"]
                pose_msg.transform.rotation = tf_data["rotation"]
                aggregated_data_msg.tf_poses.append(pose_msg)
            
            self.pub.publish(aggregated_data_msg)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = DataCollector()
        node.run()
    except rospy.ROSInterruptException:
        pass
