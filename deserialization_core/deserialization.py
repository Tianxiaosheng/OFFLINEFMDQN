import time
from deserialization_core.lon_decision_input_pb2 import *
from google.protobuf.internal.decoder import _DecodeVarint32

class PosType:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0

class PoseType:
    def __init__(self):
        self.pos = PosType()
        self.theta = 0.0

class TrajectoryType:
    def __init__(self):
        self.pose = []

class ObjInfoType:
    """
    This class is used to store the information of the object.
    """
    def __init__(self):
        self.id = 0
        self.length = 0.0
        self.width = 0.0
        self.vel = 0.0
        self.intersection_start_s = 0.0  # ego's dist to cli
        self.intersection_end_s = 0.0
        self.dist_to_intersection = 0.0 # obj's dist to cli
        self.dist_to_leave_intersection = 0.0
        self.pose = PoseType()

class ObjSetType:
    def __init__(self):
        self.decision_method = 0
        self.obj_info = []

class EgoInfoType:
    def __init__(self):
        self.vel = 0.0
        self.width = 0.0
        self.length = 0.0
        self.pose = PoseType()

class Point3dType:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0

class PolygonType:
    def __init__(self):
        self.vertex = []

class EgoExtraInfoType:
    def __init__(self):
        self.pose = PoseType()
        self.polygon = PolygonType()
        self.trajectory = TrajectoryType()

class ObjExtraInfoType:
    def __init__(self):
        self.id = 0
        self.pose = PoseType()
        self.polygon = PolygonType()
        self.trajectory = TrajectoryType()

class ExtraInfoType:
    def __init__(self):
        self.ego_extra_info = EgoExtraInfoType()
        self.local_map = 0.0
        self.obj_extra_info_set = []

class LonDecisionInputType:
    def __init__(self):
        self.ego_info = EgoInfoType()
        self.obj_set = ObjSetType()
        self.extra_info = ExtraInfoType()

"""
This class is used to deserialize the observation data from the protobuf file.
"""
class Deserialization:
    def __init__(self, file_path=None):
        self.file_path = file_path
        self.lon_decision_inputs = []
        if file_path is not None:
            self.get_lon_decision_inputs_by_deserialization(file_path)

    def get_lon_decision_inputs_by_deserialization(self, file_path):
        start_time = time.time()
        # 预分配缓冲区
        buf = bytearray(4096)  # 4KB buffer
        self.lon_decision_inputs = []

        with open(file_path, 'rb') as f:
            while True:
                # 一次性读取更多数据到缓冲区
                initial_byte = f.read(1)
                if not initial_byte:
                    break

                varint_bytes = [initial_byte[0]]
                while varint_bytes[-1] & 0x80:
                    next_byte = f.read(1)
                    if not next_byte:
                        break
                    varint_bytes.append(next_byte[0])

                size, _ = _DecodeVarint32(bytes(varint_bytes), 0)

                # 直接读取完整消息
                data = f.read(size)
                if not data:
                    break

                # 复用LonDecisionInput对象
                lon_decision_input = LonDecisionInput()
                lon_decision_input.ParseFromString(data)

                # 转换并存储
                self.lon_decision_inputs.append(
                    self.convert_lon_decision_input_to_type(lon_decision_input))

        end_time = time.time()
        print(f"Time taken to get lon decision inputs: {end_time - start_time} seconds")

    def convert_lon_decision_input_to_type(self, lon_decision_input):
        # 预创建对象,减少重复创建
        result = LonDecisionInputType()

        # 直接赋值,避免调用额外函数
        result.ego_info = self._fast_convert_ego_info(lon_decision_input.ego_info)
        result.obj_set = self._fast_convert_obj_set(lon_decision_input.obj_set) 
        result.extra_info = self._fast_convert_extra_info(lon_decision_input.extra_info)

        # 内联更新函数
        self._fast_update_obj_extra_info(result)
        self._fast_update_ego_extra_info(result)

        return result

    def _fast_convert_obj_extra_info_set(self, obj_extra_info_set):
        result = []
        for obj_extra_info in obj_extra_info_set.obj_extra_info:
            result.append(self._fast_convert_obj_extra_info(obj_extra_info))
        return result

    def _fast_convert_obj_extra_info(self, obj_extra_info):
        result = ObjExtraInfoType()
        result.id = obj_extra_info.obj_orig_id
        result.pose = obj_extra_info.obj_pose
        return result

    def _fast_convert_extra_info(self, extra_info):
        result = ExtraInfoType()
        result.ego_extra_info = self._fast_convert_ego_extra_info(extra_info.ego_extra_info)
        result.local_map = extra_info.local_map
        result.obj_extra_info_set = self._fast_convert_obj_extra_info_set(extra_info.obj_extra_info_set)
        return result

    def _fast_convert_ego_info(self, ego_info):
        result = EgoInfoType()
        result.vel = ego_info.vel
        result.width = 2.0  # 固定值
        result.length = 3.6 # 固定值
        result.pose = PoseType() # 后续会被更新
        return result

    def _fast_convert_ego_extra_info(self, ego_extra_info):
        result = EgoExtraInfoType()
        result.pose = ego_extra_info.ego_pose
        return result

    def _fast_convert_obj_set(self, obj_set):
        result = ObjSetType()
        result.obj_info = []

        # 预分配列表空间
        obj_count = len(obj_set.obj_info)
        result.obj_info = [None] * obj_count

        for i, obj in enumerate(obj_set.obj_info):
            obj_info = ObjInfoType()
            obj_info.id = obj.orig_obj_id
            obj_info.length = obj.length
            obj_info.width = obj.width
            obj_info.vel = obj.vel
            obj_info.intersection_start_s = obj.intersection_start_s
            obj_info.intersection_end_s = obj.intersection_end_s
            obj_info.dist_to_intersection = obj.dist_to_intersection
            obj_info.dist_to_leave_intersection = obj.dist_to_leave_intersection
            obj_info.pose = PoseType()
            result.obj_info[i] = obj_info

        return result

    def _fast_update_obj_extra_info(self, lon_decision_input):
        obj_info_list = lon_decision_input.obj_set.obj_info
        extra_info_list = lon_decision_input.extra_info.obj_extra_info_set

        for i in range(min(len(obj_info_list), len(extra_info_list))):
            obj_info_list[i].pose = extra_info_list[i].pose

    def _fast_update_ego_extra_info(self, lon_decision_input):
        lon_decision_input.ego_info.pose = lon_decision_input.extra_info.ego_extra_info.pose

    def get_lon_decision_inputs_size(self):
        return len(self.lon_decision_inputs)

    def get_lon_decision_input_by_frame(self, frame):
        return self.lon_decision_inputs[frame]

    def get_ego_info_by_frame(self, frame):
        return self.lon_decision_inputs[frame].ego_info

    def get_obj_set_by_frame(self, frame):
        return self.lon_decision_inputs[frame].obj_set

    def get_extra_info_by_frame(self, frame):
        return self.lon_decision_inputs[frame].extra_info

    def get_ego_extra_info_by_frame(self, frame):
        return self.lon_decision_inputs[frame].extra_info.ego_extra_info

    def get_obj_extra_info_set_by_frame(self, frame):
        return self.lon_decision_inputs[frame].extra_info.obj_extra_info_set

    def unify_heading_to_zero(self, heading, base_heading):
        return (heading - base_heading + math.pi) % (2 * math.pi) - math.pi

    def get_obj_heading_from_obj_extra_info(self, obj_extra_info):
        return obj_extra_info.pose.theta

    def get_ego_heading_from_ego_extra_info(self, ego_extra_info):
        return ego_extra_info.pose.theta

    def get_ego_heading_from_lon_decision_input(self, lon_decision_input):
        return lon_decision_input.ego_info.pose.theta

    def get_ego_position_from_lon_decision_input(self, lon_decision_input):
        return lon_decision_input.ego_info.pose.pos

    def get_ego_vertex_from_lon_decision_input(self, lon_decision_input):
        return lon_decision_input.ego_info.polygon.vertex

    def get_ego_vel_from_lon_decision_input(self, lon_decision_input):
        return lon_decision_input.ego_info.vel

    def get_ego_width_from_lon_decision_input(self, lon_decision_input):
        return lon_decision_input.ego_info.width

    def get_ego_length_from_lon_decision_input(self, lon_decision_input):
        return lon_decision_input.ego_info.length

    def get_obj_position_from_obj_info(self, obj_info):
        return obj_info.pose.pos

    def get_obj_heading_from_obj_info(self, obj_info):
        return obj_info.pose.theta

    def get_obj_vel_from_obj_info(self, obj_info):
        return obj_info.vel

    def get_obj_width_from_obj_info(self, obj_info):
        return obj_info.width

    def get_obj_length_from_obj_info(self, obj_info):
        return obj_info.length

    def get_obj_dtc_from_obj_info(self, obj_info):
        return obj_info.dist_to_intersection

    def dump_ego_info_by_frame(self, frame):
        ego_info = self.get_ego_info_by_frame(frame)
        print(f"[frame {frame}]ego_info.vel: {ego_info.vel}m/s, ego_pose: {ego_info.pose.pos.x}, {ego_info.pose.pos.y}, ego_heading: {ego_info.pose.theta}")

    def dump_obj_info_by_frame(self, frame):
        obj_set = self.get_obj_set_by_frame(frame)
        print(f"[obj_size {len(obj_set.obj_info)}]")
        i = 0
        for obj_info in obj_set.obj_info:
            print(f"obj_info[{i}].id: {obj_info.id}, vel: {obj_info.vel}m/s, dtc: {obj_info.dist_to_intersection}m, obj_pose: {obj_info.pose.pos.x}, {obj_info.pose.pos.y}, heading: {obj_info.pose.theta}")
            i += 1
