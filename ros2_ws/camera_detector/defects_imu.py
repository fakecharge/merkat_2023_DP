import rclpy
from rclpy.node import Node

import numpy as np

from std_msgs.msg import String
from sensor_msgs.msg import NavSatFix 
from geometry_msgs.msg import TwistStamped

import collections

class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        
        count_movmean_angular = 1000
        count_movmean_linear = 10000
        
        # Коэффициенты режекторного фильтра для фильтрации резонансных вибраций в спектре данных с акселерометра
        # self.filter_coeff = [
        # -0.00189917708083501, 0.0119590941903593, 
        # 0.00203721819606902, -0.000682856814161474, -0.00247282424872092, 
        # -0.00348182855600657, -0.00318068706409537, -0.00154098767599897, 
        # 0.000757136048494197, 0.00261760668227961, 0.00310397147212855,
        # 0.00197469933766154, -0.000141114960947105, -0.00203251582197552, 
        # -0.00254598494345881, -0.00127290652071412, 0.00114498400414793, 
        # 0.00328725206813075, 0.00367956770997874, 0.00168053643799527, 
        # -0.00204367892422289, -0.00568252648868998, -0.00713880246999976, 
        # -0.0051435122719487, -6.85182822795501e-05, 0.00603860927526958, 
        # 0.0102992909348262, 0.0103833655897436, 0.00574349035403689, 
        # -0.00193606492719748, -0.00937279536786627, -0.0131686644333031, 
        # -0.0114400407176885, -0.00479791225389586, 0.00384985273946854, 
        # 0.0106340392857113, 0.0125584826719726, 0.00895112747055478, 
        # 0.00178292913236561, -0.00532727769821323, -0.00893960628235905, 
        # -0.00754199978967519, -0.00231470366827828, 0.00350965732894551, 
        # 0.00638378086553483, 0.0044104946460736, -0.00150859617836363, 
        # -0.00799340929957091, -0.0108815136538694, -0.00749749463431521, 
        # 0.00167286364843449, 0.012824908751729, 0.0204013006188169,
        # 0.0197295487086119, 0.00950172128181077, -0.00716190737204408, 
        # -0.0236615977901677, -0.0326166049457297, -0.0291420948158551, 
        # -0.0132522810049325, 0.00974301544488529, 0.0311351135250161, 
        # 0.0422525009425936, 0.0380277799557732, 0.0192018734484409, 
        # -0.00778220346567391, -0.0331515597455415, -0.0474937891425834, 
        # -0.0453051909368228, -0.0270619748745168, 0.00100992652777748, 
        # 0.0291447157260082, 0.0476317740063197, 0.0500808786041518, 
        # 0.0356525785714333, 0.00923594271294042, -0.0203208069680889, 
        # -0.0431749972637371, 0.948260909320898, -0.0431749972637371, 
        # -0.0203208069680889, 0.00923594271294042, 0.0356525785714333, 
        # 0.0500808786041518, 0.0476317740063197, 0.0291447157260082, 
        # 0.00100992652777748, -0.0270619748745168, -0.0453051909368228, 
        # -0.0474937891425834, -0.0331515597455415, -0.00778220346567391, 
        # 0.0192018734484409, 0.0380277799557732, 0.0422525009425936, 
        # 0.0311351135250161, 0.00974301544488529, -0.0132522810049325, 
        # -0.0291420948158551, -0.0326166049457297, -0.0236615977901677, 
        # -0.00716190737204408, 0.00950172128181077, 0.0197295487086119, 
        # 0.0204013006188169, 0.012824908751729, 0.00167286364843449, 
        # -0.00749749463431521, -0.0108815136538694, -0.00799340929957091, 
        # -0.00150859617836363, 0.0044104946460736, 0.00638378086553483, 
        # 0.00350965732894551, -0.00231470366827828, -0.00754199978967519, 
        # -0.00893960628235905, -0.00532727769821323, 0.00178292913236561, 
        # 0.00895112747055478, 0.0125584826719726, 0.0106340392857113, 
        # 0.00384985273946854, -0.00479791225389586, -0.0114400407176885, 
        # -0.0131686644333031, -0.00937279536786627, -0.00193606492719748, 
        # 0.00574349035403689, 0.0103833655897436, 0.0102992909348262, 
        # 0.00603860927526958, -6.85182822795501e-05, -0.0051435122719487, 
        # -0.00713880246999976, -0.00568252648868998, -0.00204367892422289, 
        # 0.00168053643799527, 0.00367956770997874, 0.00328725206813075, 
        # 0.00114498400414793, -0.00127290652071412, -0.00254598494345881, 
        # -0.00203251582197552, -0.000141114960947105, 0.00197469933766154, 
        # 0.00310397147212855, 0.00261760668227961, 0.000757136048494197, 
        # -0.00154098767599897, -0.00318068706409537, -0.00348182855600657, 
        # -0.00247282424872092, -0.000682856814161474, 0.00203721819606902, 
        # 0.0119590941903593, -0.00189917708083501]
        
        self.defects_val = 0
        
        self.linear_x = collections.deque(maxlen=count_movmean_linear)
        self.linear_y = collections.deque(maxlen=count_movmean_linear)
        self.angular_x = collections.deque(maxlen=count_movmean_angular)
        self.angular_y = collections.deque(maxlen=count_movmean_angular)
        
        self.coor = [0, 0, 0]
        
        self.publisher_defects = self.create_publisher(String, 'defects_find_innino', 10)
        
        self.subscription_coor = self.create_subscription(
            NavSatFix,
            '/imu_sensor/imu/nav_sat_fix',
            self.listener_callback_coor,
            10)
        
        self.subscription_accz = self.create_subscription(
            TwistStamped,
            '/imu_sensor/imu/velocity',
            self.listener_callback_acc,
            10)
        
        self.subscription_accz # prevent unused variable warning
        self.subscription_coor # prevent unused variable warning
    
    def publisher_defects_callback(self):
        msg = String()
        if self.coor[0] > 0:
            msg.data = 'defects coord: ' + str(self.coor[0]) + ", " + str(self.coor[1]) + ", " + str(self.coor[2]) + ", val: " + str(self.defects_val)
            self.publisher_defects.publish(msg)
            self.get_logger().info('Publishing: "%s"' % msg.data)
            
    def listener_callback_coor(self, msg):
        #self.get_logger().info(f"{msg.altitude}")
        self.coor = [ msg.latitude, msg.longitude, msg.altitude ]

    def listener_callback_acc(self, msg):
        treshold = 0.1 # Порог обнаружения ямы. [g]
        self.linear_x.append(msg.twist.linear.x)
        self.linear_y.append(msg.twist.linear.y) 
        self.angular_x.append(msg.twist.angular.x) # Акселерометр ось Y (поперек движения)
        self.angular_y.append(msg.twist.angular.y) # Акселерометр ось Z
        
        motion_avg_angular = np.average(np.array(self.angular_y)) # Скользящее среднее для компенсации дрейфа нуля
        #motion_avg = np.average(data, weights=self.filter_coeff) # Фильтрация сигнала. удаляем резонансы
        self.defects_val  = abs(abs(msg.twist.angular.y) - abs(motion_avg_angular)) # "Величина ямы" в [g]
        
        # превышение ускорения по Z (попали в яму или резкий старт/останов).
        if self.defects_val > treshold:
            alarm = 1 
        else:
            alarm = 0
        
        self.defects_val = alarm * self.defects_val
            
        s_across_mean = np.average(np.array(self.angular_x)) # Скользящее среднее крена

        # Фильтрация резкого начала или окончания движения
        s_move = abs(msg.twist.linear.x) + abs(msg.twist.linear.y)
        s_move_mean = np.average(np.abs(np.array(self.linear_x)) + np.abs(np.array(self.linear_y))) # Считываем оси X + Y и вычисляем среднее в окне
        
        treshold_move = 20 # Порог начала движения. При угловой скорости по XY больше 10град/сек - резкое начало или окончание движения
        
        # Текущее значение угловой скорости больше порога. (резкое начало или окончание движения)
        if abs(s_move - s_move_mean)  > treshold_move:
            is_start_stop_move = 1    
        else:
            is_start_stop_move = 0       

        self.defects_val = self.defects_val*(1-is_start_stop_move)

        # Выход
        # defects_val == 0; % Нет ямы
        # defects_val > 0; % яма под правым колесом
        # defects_val < 0; % яма под левым колесом
        
        if msg.twist.angular.x < s_across_mean:
            self.defects_val = -self.defects_val    
                
        if  self.defects_val != 0:
            self.publisher_defects_callback()
        
def main(args=None):
    rclpy.init(args=args)

    defect_thread = MinimalSubscriber()

    rclpy.spin(defect_thread)

    defect_thread.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
    