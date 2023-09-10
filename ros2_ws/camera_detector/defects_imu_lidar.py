import rclpy
from rclpy.node import Node

import numpy as np

from std_msgs.msg import String
from sensor_msgs.msg import NavSatFix 
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

import collections

from typing import Iterable, List, NamedTuple, Optional

try:
    from numpy.lib.recfunctions import (structured_to_unstructured, unstructured_to_structured)
except ImportError:
    from sensor_msgs_py.numpy_compat import (structured_to_unstructured,
                                             unstructured_to_structured)


_DATATYPES = {}
_DATATYPES[PointField.INT8] = np.dtype(np.int8)
_DATATYPES[PointField.UINT8] = np.dtype(np.uint8)
_DATATYPES[PointField.INT16] = np.dtype(np.int16)
_DATATYPES[PointField.UINT16] = np.dtype(np.uint16)
_DATATYPES[PointField.INT32] = np.dtype(np.int32)
_DATATYPES[PointField.UINT32] = np.dtype(np.uint32)
_DATATYPES[PointField.FLOAT32] = np.dtype(np.float32)
_DATATYPES[PointField.FLOAT64] = np.dtype(np.float64)

DUMMY_FIELD_PREFIX = 'unnamed_field'


def read_points(
        cloud: PointCloud2,
        field_names: Optional[List[str]] = None,
        skip_nans: bool = False,
        uvs: Optional[Iterable] = None,
        reshape_organized_cloud: bool = False) -> np.ndarray:
    """
    Read points from a sensor_msgs.PointCloud2 message.

    :param cloud: The point cloud to read from sensor_msgs.PointCloud2.
    :param field_names: The names of fields to read. If None, read all fields.
                        (Type: Iterable, Default: None)
    :param skip_nans: If True, then don't return any point with a NaN value.
                      (Type: Bool, Default: False)
    :param uvs: If specified, then only return the points at the given
        coordinates. (Type: Iterable, Default: None)
    :param reshape_organized_cloud: Returns the array as an 2D organized point cloud if set.
    :return: Structured NumPy array containing all points.
    """
    assert isinstance(cloud, PointCloud2), \
        'Cloud is not a sensor_msgs.msg.PointCloud2'

    # Cast bytes to numpy array
    points = np.ndarray(
        shape=(cloud.width * cloud.height, ),
        dtype=dtype_from_fields(cloud.fields, point_step=cloud.point_step),
        buffer=cloud.data)

    # Keep only the requested fields
    if field_names is not None:
        assert all(field_name in points.dtype.names for field_name in field_names), \
            'Requests field is not in the fields of the PointCloud!'
        # Mask fields
        points = points[list(field_names)]

    # Swap array if byte order does not match
    if bool(sys.byteorder != 'little') != bool(cloud.is_bigendian):
        points = points.byteswap(inplace=True)

    # Check if we want to drop points with nan values
    if skip_nans and not cloud.is_dense:
        # Init mask which selects all points
        not_nan_mask = np.ones(len(points), dtype=bool)
        for field_name in points.dtype.names:
            # Only keep points without any non values in the mask
            not_nan_mask = np.logical_and(
                not_nan_mask, ~np.isnan(points[field_name]))
        # Select these points
        points = points[not_nan_mask]

    # Select points indexed by the uvs field
    if uvs is not None:
        # Don't convert to numpy array if it is already one
        if not isinstance(uvs, np.ndarray):
            uvs = np.fromiter(uvs, int)
        # Index requested points
        points = points[uvs]

    # Cast into 2d array if cloud is 'organized'
    if reshape_organized_cloud and cloud.height > 1:
        points = points.reshape(cloud.width, cloud.height)

    return points


def read_points_numpy(
        cloud: PointCloud2,
        field_names: Optional[List[str]] = None,
        skip_nans: bool = False,
        uvs: Optional[Iterable] = None,
        reshape_organized_cloud: bool = False) -> np.ndarray:
    """
    Read equally typed fields from sensor_msgs.PointCloud2 message as a unstructured numpy array.

    This method is better suited if one wants to perform math operations
    on e.g. all x,y,z fields.
    But it is limited to fields with the same dtype as unstructured numpy arrays
    only contain one dtype.

    :param cloud: The point cloud to read from sensor_msgs.PointCloud2.
    :param field_names: The names of fields to read. If None, read all fields.
                        (Type: Iterable, Default: None)
    :param skip_nans: If True, then don't return any point with a NaN value.
                      (Type: Bool, Default: False)
    :param uvs: If specified, then only return the points at the given
        coordinates. (Type: Iterable, Default: None)
    :param reshape_organized_cloud: Returns the array as an 2D organized point cloud if set.
    :return: Numpy array containing all points.
    """
    assert all(cloud.fields[0].datatype == field.datatype for field in cloud.fields[1:]
               if field_names is None or field.name in field_names), \
        'All fields need to have the same datatype. Use `read_points()` otherwise.'
    structured_numpy_array = read_points(
        cloud, field_names, skip_nans, uvs, reshape_organized_cloud)
    return structured_to_unstructured(structured_numpy_array)


def read_points_list(
        cloud: PointCloud2,
        field_names: Optional[List[str]] = None,
        skip_nans: bool = False,
        uvs: Optional[Iterable] = None) -> List[NamedTuple]:
    """
    Read points from a sensor_msgs.PointCloud2 message.

    This function returns a list of namedtuples. It operates on top of the
    read_points method. For more efficient access use read_points directly.

    :param cloud: The point cloud to read from. (Type: sensor_msgs.PointCloud2)
    :param field_names: The names of fields to read. If None, read all fields.
                        (Type: Iterable, Default: None)
    :param skip_nans: If True, then don't return any point with a NaN value.
                      (Type: Bool, Default: False)
    :param uvs: If specified, then only return the points at the given
                coordinates. (Type: Iterable, Default: None]
    :return: List of namedtuples containing the values for each point
    """
    assert isinstance(cloud, PointCloud2), \
        'cloud is not a sensor_msgs.msg.PointCloud2'

    if field_names is None:
        field_names = [f.name for f in cloud.fields]

    Point = namedtuple('Point', field_names)

    return [Point._make(p) for p in read_points(cloud, field_names,
                                                skip_nans, uvs)]


def dtype_from_fields(fields: Iterable[PointField], point_step: Optional[int] = None) -> np.dtype:
    """
    Convert a Iterable of sensor_msgs.msg.PointField messages to a np.dtype.

    :param fields: The point cloud fields.
                   (Type: iterable of sensor_msgs.msg.PointField)
    :param point_step: Point step size in bytes. Calculated from the given fields by default.
                       (Type: optional of integer)
    :returns: NumPy datatype
    """
    # Create a lists containing the names, offsets and datatypes of all fields
    field_names = []
    field_offsets = []
    field_datatypes = []
    for i, field in enumerate(fields):
        # Datatype as numpy datatype
        datatype = _DATATYPES[field.datatype]
        # Name field
        if field.name == '':
            name = f'{DUMMY_FIELD_PREFIX}_{i}'
        else:
            name = field.name
        # Handle fields with count > 1 by creating subfields with a suffix consiting
        # of "_" followed by the subfield counter [0 -> (count - 1)]
        assert field.count > 0, "Can't process fields with count = 0."
        for a in range(field.count):
            # Add suffix if we have multiple subfields
            if field.count > 1:
                subfield_name = f'{name}_{a}'
            else:
                subfield_name = name
            assert subfield_name not in field_names, 'Duplicate field names are not allowed!'
            field_names.append(subfield_name)
            # Create new offset that includes subfields
            field_offsets.append(field.offset + a * datatype.itemsize)
            field_datatypes.append(datatype.str)

    # Create dtype
    dtype_dict = {
            'names': field_names,
            'formats': field_datatypes,
            'offsets': field_offsets
    }
    if point_step is not None:
        dtype_dict['itemsize'] = point_step
    return np.dtype(dtype_dict)


def create_cloud(
        header: Header,
        fields: Iterable[PointField],
        points: Iterable,
        point_step: Optional[int] = None) -> PointCloud2:
    """
    Create a sensor_msgs.msg.PointCloud2 message.

    :param header: The point cloud header. (Type: std_msgs.msg.Header)
    :param fields: The point cloud fields.
                   (Type: iterable of sensor_msgs.msg.PointField)
    :param points: The point cloud points. List of iterables, i.e. one iterable
                   for each point, with the elements of each iterable being the
                   values of the fields for that point (in the same order as
                   the fields parameter)
    :param point_step: Point step size in bytes. Calculated from the given fields by default.
                       (Type: optional of integer)
    :return: The point cloud as sensor_msgs.msg.PointCloud2
    """
    # Check if input is numpy array
    if isinstance(points, np.ndarray):
        # Check if this is an unstructured array
        if points.dtype.names is None:
            assert all(fields[0].datatype == field.datatype for field in fields[1:]), \
                'All fields need to have the same datatype. Pass a structured NumPy array \
                    with multiple dtypes otherwise.'
            # Convert unstructured to structured array
            points = unstructured_to_structured(
                points,
                dtype=dtype_from_fields(fields, point_step))
        else:
            assert points.dtype == dtype_from_fields(fields, point_step), \
                'PointFields and structured NumPy array dtype do not match for all fields! \
                    Check their field order, names and types.'
    else:
        # Cast python objects to structured NumPy array (slow)
        points = np.array(
            # Points need to be tuples in the structured array
            list(map(tuple, points)),
            dtype=dtype_from_fields(fields, point_step))

    # Handle organized clouds
    assert len(points.shape) <= 2, \
        'Too many dimensions for organized cloud! \
            Points can only be organized in max. two dimensional space'
    height = 1
    width = points.shape[0]
    # Check if input points are an organized cloud (2D array of points)
    if len(points.shape) == 2:
        height = points.shape[1]

    # Convert numpy points to array.array
    memory_view = memoryview(points)
    casted = memory_view.cast('B')
    array_array = array.array('B')
    array_array.frombytes(casted)

    # Put everything together
    cloud = PointCloud2(
        header=header,
        height=height,
        width=width,
        is_dense=False,
        is_bigendian=sys.byteorder != 'little',
        fields=fields,
        point_step=points.dtype.itemsize,
        row_step=(points.dtype.itemsize * width))
    # Set cloud via property instead of the constructor because of the bug described in
    # https://github.com/ros2/common_interfaces/issues/176
    cloud.data = array_array
    return cloud


def create_cloud_xyz32(header: Header, points: Iterable) -> PointCloud2:
    """
    Create a sensor_msgs.msg.PointCloud2 message with (x, y, z) fields.

    :param header: The point cloud header. (Type: std_msgs.msg.Header)
    :param points: The point cloud points. (Type: Iterable)
    :return: The point cloud as sensor_msgs.msg.PointCloud2.
    """
    fields = [PointField(name='x', offset=0,
                         datatype=PointField.FLOAT32, count=1),
              PointField(name='y', offset=4,
                         datatype=PointField.FLOAT32, count=1),
              PointField(name='z', offset=8,
                         datatype=PointField.FLOAT32, count=1)]
    return create_cloud(header, fields, points)
import sys

if sys.version_info >= (3, 0):
    def xrange(*args, **kwargs):
        return iter(range(*args, **kwargs))
    

class Ros2InNINOSubscriber(Node):

    def __init__(self):
        super().__init__('innino_subscriber')
        
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
        
        self.lidar_x = 0
        self.lidar_y = 0
        
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
        
        self.subscription_lidar = self.create_subscription(
            PointCloud2,
            '/points',
            self.listener_callback_lidar,
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

    def listener_callback_lidar(self, msg):
        W = 1024 # Ширина скана
        H = 64 # Дальность скана
        w = 100 # Ширина анализируемой области
        h = 35 # Высота анализируемой области
        min_az = int(np.round(W/2-w/2))
        max_az = int(np.round(W/2+w/2))
        min_D = 5
        max_D = min_D+h
        
        th = 0.35 
        
        gen = read_points(msg, skip_nans=True, field_names=("x", "y", "z"))#, uvs=self.points2D)
        fff = [ x for x, y, z in gen]
        
        points3D_x = np.array([point[0] for point in gen]).reshape(64, 1024)
        points3D_y = np.array([point[1] for point in gen]).reshape(64, 1024)
        points3D_z = np.array([point[2] for point in gen]).reshape(64, 1024)
        
        points3D_x = points3D_x[min_D:max_D, min_az:max_az]
        points3D_y = points3D_y[min_D:max_D, min_az:max_az]
        points3D_z = points3D_z[min_D:max_D, min_az:max_az]
        
        def running_mean(x, N):
            m, n = x.shape
            xx = np.convolve(x.reshape(x.size), np.ones(N), 'same') / N
            return xx.reshape(m, n)
        
        rx = np.diff(points3D_x, axis=1)
        ry = np.diff(points3D_y, axis=1)
        
        r = np.sqrt(rx*rx + ry*ry)
        r[r > 0.5] = 0
        
        m, n = r.shape
        r_mean = np.mean(r, axis=1).reshape(m,1)
        #r_mean = running_mean(r, 1000)
       
        detect = np.abs(r_mean * np.ones(r.shape) - r) > th
        #detect = r > 2 * r_mean * np.ones(r.shape)
        
        # Координаты обнаруженного дефекта относительно носителя
        self.lidar_x = points3D_x[np.nonzero(detect)] 
        self.lidar_y = points3D_y[np.nonzero(detect)]

        #if self.lidar_x.size > 0 and self.lidar_y.size > 0:
        #    self.publisher_defects_callback()
    
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

    defect_thread = Ros2InNINOSubscriber()

    rclpy.spin(defect_thread)

    defect_thread.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
