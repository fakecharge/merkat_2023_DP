import sqlite3
from rosidl_runtime_py.convert import message_to_csv 
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
import numpy as np

import matplotlib.pyplot as plt

class BagFileParser():
    def __init__(self, bag_file):
        self.conn = sqlite3.connect(bag_file)
        self.cursor = self.conn.cursor()

        ## create a message type map
        #topics_data = self.cursor.execute("SELECT id, name, type FROM topics").fetchall()
        #print(topics_data)
        topics_data = [(7, '/imu_sensor/imu/velocity', 'geometry_msgs/msg/TwistStamped'), (8, '/imu_sensor/imu/pos_ecef', 'geometry_msgs/msg/PointStamped'), (9, '/imu_sensor/imu/odometry', 'nav_msgs/msg/Odometry'), (10, '/imu_sensor/imu/utc_ref', 'sensor_msgs/msg/TimeReference'),(11, '/imu_sensor/imu/nav_sat_fix', 'sensor_msgs/msg/NavSatFix'),  (13, '/imu_sensor/imu/data', 'sensor_msgs/msg/Imu'), (14, '/points', 'sensor_msgs/msg/PointCloud2')]
        self.topic_type = {name_of:type_of for id_of,name_of,type_of in topics_data}
        self.topic_id = {name_of:id_of for id_of,name_of,type_of in topics_data}
        print(self.topic_type )
        print(self.topic_id)
        self.topic_msg_message = {name_of: get_message(type_of) for id_of,name_of,type_of in topics_data}

    def __del__(self):
        self.conn.close()

    # Return [(timestamp0, message0), (timestamp1, message1), ...]
    def get_messages(self, topic_name):

        topic_id = self.topic_id[topic_name]
        # Get from the db
        rows = self.cursor.execute("SELECT timestamp, data FROM messages WHERE topic_id = {}".format(topic_id)).fetchall()
        #rows = self.cursor.execute("SELECT timestamp, data FROM messages WHERE topic_id = 7")
        # Deserialise all and timestamp them

        return [ (timestamp,deserialize_message(data, self.topic_msg_message[topic_name])) for timestamp, data in rows]

    def save_messages(self, topic_name):
        topic_id = self.topic_id[topic_name]
        # Get from the db
        rows = self.cursor.execute("SELECT timestamp, data FROM messages WHERE topic_id = {}".format(topic_id)).fetchall()
        #rows = self.cursor.execute("SELECT timestamp, data FROM messages WHERE topic_id = 7")
        # Deserialise all and timestamp them

        return [ (timestamp, message_to_csv(deserialize_message(data, self.topic_msg_message[topic_name]))) for timestamp, data in rows]


if __name__ == "__main__":

        bag_file = 'rosbag2_2023_09_04-11_56_58_0.db3'

        parser = BagFileParser(bag_file)
        count = 1
        
        name = 'points'
        data_name = '/points'
        #trajectory = parser.get_messages(data_name)[0][1]
        #print(trajectory)
        data = parser.save_messages(data_name)
        f = open(name + '.csv', 'w')
        for t, s in data:
            f.write(str(t) + ', ' + s +'\n')
        #np.savetxt('data.csv', str, delimiter=',')
        f.close()
        #print(type(trajectory))
        #print(trajectory)
        #p_des_1 = [trajectory.points[i].positions[0] for i in range(len(trajectory.points)[:count])]
        #t_des = [trajectory.points[i].time_from_start.sec + trajectory.points[i].time_from_start.nanosec*1e-9 for i in range(len(trajectory.points)[:count])]

        # actual = parser.get_messages("/points")

        #plt.plot(t_des, p_des_1)
        #plt.show()
