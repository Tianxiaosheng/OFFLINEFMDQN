import math
import numpy as np

def cvt_pose_local_to_global(local_pos_x, local_pos_y, local_pos_theta,
                             base_pos_x, base_pos_y, base_pos_theta):
    global_pos_x = base_pos_x
    global_pos_x += math.sin(base_pos_theta) * local_pos_x + math.cos(base_pos_theta) * local_pos_y
    global_pos_y = base_pos_y
    global_pos_y += math.sin(base_pos_theta) * local_pos_y - math.cos(base_pos_theta) * local_pos_x;

    global_pos_theta = local_pos_theta + base_pos_theta - math.pi / 2.0

    return (global_pos_x, global_pos_y, global_pos_theta)

def cvt_pose_global_to_local(global_pos_x, global_pos_y, global_pos_theta,\
                             base_pos_x, base_pos_y, base_pos_theta):
    dx = global_pos_x - base_pos_x
    dy = global_pos_y - base_pos_y
    theta = base_pos_theta

    local_pos_x = math.sin(theta) * dx - math.cos(theta) * dy
    local_pos_y = math.cos(theta) * dx + math.sin(theta) * dy

    local_pos_theta = math.pi / 2.0 + global_pos_theta - theta
    if local_pos_theta < 0.0:
        local_pos_theta += math.pi * 2.0
    local_pos_theta = local_pos_theta % (2.0 * math.pi)
    return (local_pos_x, local_pos_y, local_pos_theta)

def is_point_in_polygon(point, polygon):
    """
    Check if a point is inside a polygon.

    :param point: Tuple (x, y) representing the point
    :param polygon: List of tuples representing the polygon's corners
    :return: True if the point is inside the polygon, False otherwise
    """
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        # Check if the point is on a boundary
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):  
            inside = not inside  

        j = i

    return inside

class OccupancyGrid:
    def __init__(self, shape_dim, delta_x, delta_y, render):
        self.shape_dim = shape_dim
        self.channels = shape_dim[0]
        self.height = shape_dim[1]  # (channels, hight, width)
        self.width = shape_dim[2]
        self.delta_x = delta_x
        self.delta_y = delta_y
        self.render = render
        #  grid's global pose of coordinate origin
        self.center_x = 0.0
        self.center_y = 0.0
        self.heading = 0.0

    def preprocess_occupancy_grid(self, ego_center, ego_heading):
        #  grid's global pose of coordinate origin
        dist = (self.width - 1) / 2.0 * self.delta_x
        self.center_x = ego_center[0] - dist * math.sin(ego_heading)
        self.center_y = ego_center[1] + dist * math.cos(ego_heading)
        self.heading = ego_heading % (2.0 * math.pi)

        # self.grid = np.zeros(self.shape_dim, dtype=float)
        self.grid = np.full(self.shape_dim, -1.0, dtype=float)

        if self.render:
            print("ego_x:{}, y:{}, th:{}, grid_global_x:{}, y:{}, th:{}".\
                  format(ego_center[0], ego_center[1], ego_heading, \
                         self.center_x, self.center_y, self.heading))

    def update_occupancy_grid(self, obj_center, obj_heading, obj_vel, bounding_box, obj_dist_to_cli):
        # Convert global pose of obj to local pose of grid
        obj_center_x, obj_center_y, heading_local = \
                cvt_pose_global_to_local(obj_center[0], obj_center[1], obj_heading,\
                                         self.center_x, self.center_y, self.heading)

        if self.render:
            print("obj_global->x{}, y{}, th{}, length{}, width{}, local->x{}, y{}, th{}".\
                  format(obj_center[0], obj_center[1], obj_heading, bounding_box[0], bounding_box[1],\
                         obj_center_x, obj_center_y, heading_local))

        feature = [1.0, heading_local, obj_vel, obj_dist_to_cli]

        # Calculate the rotation matrix
        cos_theta = math.cos(heading_local)
        sin_theta = math.sin(heading_local)
        rotation_matrix = [[cos_theta, -sin_theta], [sin_theta, cos_theta]]

        # Calculate the corners of the rectangle in local coordinates (centered at (0, 0))  
        half_height = bounding_box[0] / 2.0
        half_width = bounding_box[1] / 2.0
        corners = [
            (-half_height, -half_width),
            (-half_height, half_width),
            (half_height, half_width),
            (half_height, -half_width)
        ]

        # Rotate and translate the corners to global coordinates
        rotated_corners = []
        for corner in corners:
            # Rotate
            rotated_x, rotated_y = (rotation_matrix[0][0]*corner[0] + rotation_matrix[0][1]*corner[1],
                                    rotation_matrix[1][0]*corner[0] + rotation_matrix[1][1]*corner[1])
            # Translate
            rotated_corners.append((rotated_x + obj_center_x, rotated_y + obj_center_y))

        # Get the minimum and maximum x and y coordinates to define the bounding box
        x_min, y_min = min(c[0] for c in rotated_corners), min(c[1] for c in rotated_corners)
        x_max, y_max = max(c[0] for c in rotated_corners), max(c[1] for c in rotated_corners)

        # Update the occupancy grid
        for x in range(int(x_min // self.delta_x), int(x_max // self.delta_x) + 1):
            for y in range(int(y_min // self.delta_y), int(y_max // self.delta_y) + 1):
                if x < 0 or x >= self.width or y < 0 or y >= self.height:
                    continue
                # Check if the current cell is within the rotated rectangle
                if is_point_in_polygon((x * self.delta_x, y * self.delta_y), rotated_corners):
                    # feature [occupancy_status, obj_heading, obj_speeding, dist_to_cli]
                    for ch in range(self.channels):
                        self.grid[ch][y][x] = feature[ch]

    def dump_ogm_graph(self, grid, channel):
        if channel == 0:
            ch = 'obj_occupancy'
        elif channel == 1:
            ch = 'obj_heading'
        elif channel == 2:
            ch = 'obj_speed'
        elif channel == 3:
            ch = 'DTC'
        else:
            ch = 'None'

        print('------------------------{}---------------------------'.format(ch))
        for row in reversed(grid[channel]):
            print(' '.join(str(int(cell)) for cell in row))

    def dump_ogm_graphs(self, grid):
        for channel in range(self.channels):
            self.dump_ogm_graph(grid, channel)