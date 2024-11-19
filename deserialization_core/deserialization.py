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
    def __init__(self):
        self.data_file = None
        self.lon_decision_inputs = []

    def get_lon_decision_inputs_by_deserialization(self, file_path):
        with open(file_path, 'rb') as f:
            while True:
                # 读取消息长度的前几个字节
                buf = f.read(1)
                if not buf:
                    break
                # 继续读取直到完整的 varint 被读取
                while buf[-1] & 0x80:
                    next_byte = f.read(1)
                    if not next_byte:
                        break
                    buf += next_byte

                # 如果没有读取到完整的 varint，退出循环
                if not buf:
                    break

                # 解码消息长度
                size, pos = _DecodeVarint32(buf, 0)

                # 读取消息内容
                data = f.read(size)
                if not data:
                    break

                # 反序列化消息
                lon_decision_input = LonDecisionInput()
                lon_decision_input.ParseFromString(data)
                lon_decision_input_type = self.convert_lon_decision_input_to_type(lon_decision_input)
                self.lon_decision_inputs.append(lon_decision_input_type)

    def convert_ego_info_to_type(self, ego_info):
        ego_info_type = EgoInfoType()
        ego_info_type.vel = ego_info.vel
        return ego_info_type

    def convert_obj_info_to_type(self, obj_info):
        obj_info_type = ObjInfoType()
        obj_info_type.id = obj_info.orig_obj_id
        obj_info_type.length = obj_info.length
        obj_info_type.width = obj_info.width
        obj_info_type.vel = obj_info.vel
        obj_info_type.intersection_start_s = obj_info.intersection_start_s
        obj_info_type.intersection_end_s = obj_info.intersection_end_s
        obj_info_type.dist_to_intersection = obj_info.dist_to_intersection
        obj_info_type.dist_to_leave_intersection = obj_info.dist_to_leave_intersection
        return obj_info_type

    def convert_obj_set_to_type(self, obj_set):
        obj_set_type = ObjSetType()
        for obj_info in obj_set.obj_info:
            obj_info_type = self.convert_obj_info_to_type(obj_info)
            obj_set_type.obj_info.append(obj_info_type)

        return obj_set_type

    def convert_pose_to_type(self, pose):
        pose_type = PoseType()
        pose_type.pos.x = pose.pos.x
        pose_type.pos.y = pose.pos.y
        pose_type.theta = pose.theta
        return pose_type

    def convert_polygon_to_type(self, polygon):
        polygon_type = PolygonType()
        polygon_type.vertex = polygon.vertex
        return polygon_type

    def convert_trajectory_to_type(self, trajectory):
        trajectory_type = TrajectoryType()
        for pose in trajectory.pose:
            pose_type = self.convert_pose_to_type(pose)
            trajectory_type.pose.append(pose_type)
        return trajectory_type

    def convert_ego_extra_info_to_type(self, ego_extra_info):
        ego_extra_info_type = EgoExtraInfoType()
        ego_extra_info_type.pose = self.convert_pose_to_type(ego_extra_info.ego_pose)
        ego_extra_info_type.polygon = self.convert_polygon_to_type(ego_extra_info.ego_polygon)
        ego_extra_info_type.trajectory = self.convert_trajectory_to_type(ego_extra_info.ego_traj)
        return ego_extra_info_type

    def convert_obj_extra_info_to_type(self, obj_extra_info):
        obj_extra_info_type = ObjExtraInfoType()
        obj_extra_info_type.id = obj_extra_info.obj_orig_id
        obj_extra_info_type.pose = self.convert_pose_to_type(obj_extra_info.obj_pose)
        obj_extra_info_type.polygon = self.convert_polygon_to_type(obj_extra_info.obj_polygon)
        obj_extra_info_type.trajectory = self.convert_trajectory_to_type(obj_extra_info.obj_traj)
        return obj_extra_info_type

    def convert_extra_info_to_type(self, extra_info):
        extra_info_type = ExtraInfoType()
        extra_info_type.ego_extra_info = self.convert_ego_extra_info_to_type(extra_info.ego_extra_info)
        for obj_extra_info in extra_info.obj_extra_info_set.obj_extra_info:
            obj_extra_info_type = self.convert_obj_extra_info_to_type(obj_extra_info)
            extra_info_type.obj_extra_info_set.append(obj_extra_info_type)
        return extra_info_type

    def update_obj_extra_info_to_obj_info(self, lon_decision_input_type):
        for i, obj_info in enumerate(lon_decision_input_type.obj_set.obj_info):
            if i >= len(lon_decision_input_type.extra_info.obj_extra_info_set):
                break
            obj_extra_info = lon_decision_input_type.extra_info.obj_extra_info_set[i]
            obj_info.pose = obj_extra_info.pose

    def update_ego_extra_info_to_ego_info(self, lon_decision_input_type):
        lon_decision_input_type.ego_info.pose = lon_decision_input_type.extra_info.ego_extra_info.pose

    def convert_lon_decision_input_to_type(self, lon_decision_input):
        lon_decision_input_type = LonDecisionInputType()
        lon_decision_input_type.ego_info = self.convert_ego_info_to_type(lon_decision_input.ego_info)
        lon_decision_input_type.obj_set = self.convert_obj_set_to_type(lon_decision_input.obj_set)
        lon_decision_input_type.extra_info = self.convert_extra_info_to_type(lon_decision_input.extra_info)

        self.update_obj_extra_info_to_obj_info(lon_decision_input_type)
        self.update_ego_extra_info_to_ego_info(lon_decision_input_type)

        return lon_decision_input_type

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
