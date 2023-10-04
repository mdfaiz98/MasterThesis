#!/usr/bin/env python2.7

import rospy
from e4_msgs.msg import Float32WithHeader, Float32MultiArrayWithHeader
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import Pose, Point, Quaternion
from e4_msgs.msg import AggregatedData
from collections import OrderedDict

CHILD_FRAME_ORDER = [
    "faizan_upper_Hip_0",
    "faizan_upper_Ab_0",
    "faizan_upper_Chest_0",
    "faizan_upper_Neck_0",
    "faizan_upper_Head_0",
    "faizan_upper_LShoulder_0",
    "faizan_upper_LUArm_0",
    "faizan_upper_LFArm_0",
    "faizan_upper_LHand_0",
    "faizan_upper_RShoulder_0",
    "faizan_upper_RUArm_0",
    "faizan_upper_RFArm_0",
    "faizan_upper_RHand_0"
]

class DataCollector:
    def __init__(self):
        rospy.init_node("data_collector_node", anonymous=True)

        self.pub = rospy.Publisher("/aggregated_data", AggregatedData, queue_size=10)  

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
        self.latest_data["/tf"] = {child_frame_id: [0, 0, 0, 0, 0, 0, 1] for child_frame_id in CHILD_FRAME_ORDER}
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
            received_tf_data = {tf.child_frame_id: tf for tf in data.transforms}

            for child_frame_id, tf in received_tf_data.items():
                if child_frame_id in self.latest_data["/tf"]:
                    self.latest_data["/tf"][child_frame_id] = [
                        tf.transform.translation.x,
                        tf.transform.translation.y,
                        tf.transform.translation.z,
                        tf.transform.rotation.x,
                        tf.transform.rotation.y,
                        tf.transform.rotation.z,
                        tf.transform.rotation.w
                    ]
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
            
            # Populate tf_poses with Pose objects
            for child_frame_id in CHILD_FRAME_ORDER:
                pose_list = self.latest_data["/tf"].get(child_frame_id, [0, 0, 0, 0, 0, 0, 1])
                pose_msg = Pose()
                pose_msg.position = Point(*pose_list[:3])
                pose_msg.orientation = Quaternion(*pose_list[3:])
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
