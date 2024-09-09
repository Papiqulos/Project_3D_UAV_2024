import numpy as np
import open3d as o3d
import numpy as np
from vvrpywork.constants import Color
from vvrpywork.shapes import (
    Cuboid3D, Mesh3D, Point3D, LineSet3D, Line3D
)
from itertools import combinations


def rSubset(arr, r):
    '''return list of all subsets of length r''' 
    return list(combinations(arr, r))

# Contruct a default plane pointing in the upward y direction 
def default_plane(size=1.0):
    # Define vertices of the plane (two triangles)
    vertices = np.array([[-size,  0.0,  -size],  # Vertex 0
                        [ size,  0.0,  -size],  # Vertex 1
                        [ size,  0.0,   size],  # Vertex 2
                        [-size,  0.0,   size]]) # Vertex 3

    # Define faces of the plane (two triangles)
    faces = np.array([[0, 1, 2],  # Triangle 1 (vertices 0-1-2)
                    [0, 2, 3]]) # Triangle 2 (vertices 0-2-3)

    # Create Open3D TriangleMesh object
    plane_mesh = o3d.geometry.TriangleMesh()
    plane_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    plane_mesh.triangles = o3d.utility.Vector3iVector(faces)

    # print(f"----{ len(vertices[faces]) }")

    return plane_mesh

# Generate a plane with given position and direction
def generate_plane(direction, translation, size=1.0):
    """
    Generate a plane mesh with specified orientation and position.

    Parameters:
    - direction: Direction vector [dx, dy, dz] of the plane.
    - translation: Translation vector [tx, ty, tz] for position.

    Returns:
    - plane: Open3D TriangleMesh object representing the plane.
    - transformed_center: Transformed center of the plane after applying rotation and translation.
    - transformed_dir: Transformed direction vector of the plane after applying rotation.
    """
    # Create a default plane
    plane = default_plane(size)

    # Initial center and direction of the plane
    center = np.array([0, 0, 0])
    dir = np.array([0, 1, 0])  # Default direction is along y-axis

    # Get vertices of the default plane
    vertices = np.asarray(plane.vertices)
    direction = np.array(direction)

    # Compute rotation matrix so that dir aligns with direction

    # Check if the direction is along the  negative y-axis
    if np.all(direction == np.array([0, -1, 0])):
        rotation_matrix = np.eye(3)
    # Check if the direction is along the positive y-axis
    elif not (direction==dir).all():
        dir = dir / np.linalg.norm(dir)
        direction = direction / np.linalg.norm(direction)
        axis = np.cross(dir, direction)
        angle = np.arccos(np.dot(dir, direction))
        rotation_matrix = rotation_matrix_from_axis_angle(axis, angle)
    else:
        rotation_matrix = np.eye(3)

    # Apply rigid transformation to vertices
    transformed_vertices = vertices @ rotation_matrix.T + translation 
    plane.vertices = o3d.utility.Vector3dVector(transformed_vertices)

    # Transform the center of the plane
    transformed_center = center @ rotation_matrix.T + translation

    # Transform the direction vector of the plane
    transformed_dir = dir @ rotation_matrix.T

    plane = o3d_to_mesh(plane)

    return plane, transformed_center, transformed_dir

def get_convex_hull_of_pcd(points):
    '''Creates a triangle mesh from a set of points by creating the convex hull of the points.'''
    # Create Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    hull, _ = pcd.compute_convex_hull()
    hull = o3d_to_mesh(hull)
    
    return hull

def check_point_in_cuboid(point, cuboid: Cuboid3D) -> bool:
    """
    Check if a point is inside a cuboid.

    Parameters:
    - point: Point [x, y, z] to check.
    - cuboid: Cuboid3D object representing the cuboid.

    Returns:
    - True if the point is inside the cuboid, False otherwise.
    """
    # Check if the point is inside the cuboid
    if (cuboid.x_min <= point[0] <= cuboid.x_max and
        cuboid.y_min <= point[1] <= cuboid.y_max and
        cuboid.z_min<= point[2] <= cuboid.z_max):
        return True
    else:
        return False

def intersection_of_three_planes(plane1, plane2, plane3):
    # Extract coefficients from the plane equations
    A = np.array([plane1[:3], plane2[:3], plane3[:3]])
    b = np.array([-plane1[3], -plane2[3], -plane3[3]])
    
    # Check if the determinant of A is non-zero
    if np.linalg.det(A) == 0:
        return
    
    # Solve the system of equations
    intersection_point = np.linalg.solve(A, b)
    
    return intersection_point

def remove_duplicates(points):
    '''Removes duplicate points from a set of points.'''
    return np.unique(points, axis=0)

def rotation_matrix_from_axis_angle(axis, angle):
    # Normalize the rotation axis
    axis = axis / np.linalg.norm(axis)
    
    # Components of the axis
    x, y, z = axis
    
    # Skew-symmetric matrix for the axis
    K = np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]
    ])
    
    # Identity matrix
    I = np.eye(3)
    
    # Rodrigues' rotation formula
    R = I + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    
    return R

def euler_angles_to_rotation_matrix(euler_angles):
    """
    Create a rotation matrix from Euler angles.

    Parameters:
    - euler_angles: Euler angles [rx, ry, rz] in radians

    Returns:
    - Rotation matrix
    """
    rx, ry, rz = euler_angles

    rotation_matrix_x = np.array([[1, 0, 0],
                                   [0, np.cos(rx), -np.sin(rx)],
                                   [0, np.sin(rx), np.cos(rx)]])

    rotation_matrix_y = np.array([[np.cos(ry), 0, np.sin(ry)],
                                   [0, 1, 0],
                                   [-np.sin(ry), 0, np.cos(ry)]])

    rotation_matrix_z = np.array([[np.cos(rz), -np.sin(rz), 0],
                                   [np.sin(rz), np.cos(rz), 0],
                                   [0, 0, 1]])
    # Combine rotation matrices
    rotation_matrix = rotation_matrix_z @ rotation_matrix_y @ rotation_matrix_x
    
    return rotation_matrix

def compute_euler_angles(direction):
    x, y, z = direction
    
    # Yaw (alpha)
    alpha = np.arctan2(y, x)
    
    # Pitch (beta)
    beta = np.arctan2(z, np.sqrt(x**2 + y**2))
    
    # Roll (gamma) is typically 0 for aligning vectors
    gamma = 0.0
    
    return np.array([alpha, beta, gamma])

def o3d_to_mesh(o3d_mesh:o3d.geometry.TriangleMesh) -> Mesh3D:
        '''Converts an Open3D mesh to a Mesh3D object.

        Args:
            o3d_mesh: The Open3D mesh

        Returns:
            mesh: The Mesh3D object
        '''
        mesh = Mesh3D()
        mesh.vertices = np.array(o3d_mesh.vertices)
        mesh.triangles = np.array(o3d_mesh.triangles)

        return mesh
    
def mesh_to_o3d(mesh:Mesh3D) -> o3d.geometry.TriangleMesh:
    '''Converts a Mesh3D object to an Open3D mesh.

    Args:
        mesh: The Mesh3D object

    Returns:
        o3d_mesh: The Open3D mesh
    '''
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.triangles)

    return o3d_mesh

# Not used
def find_edge_vertices_of_mesh(mesh:Mesh3D) -> tuple:
    '''Finds the vertices of the mesh that are at the edges of the mesh.

    Args:
        mesh: The Mesh3D object

    Returns:
        max_x: The vertex of the mesh with the maximum x-coordinate
        max_y: The vertex of the mesh with the maximum y-coordinate
        max_z: The vertex of the mesh with the maximum z-coordinate
        min_x: The vertex of the mesh with the minimum x-coordinate
        min_y: The vertex of the mesh with the minimum y-coordinate
        min_z: The vertex of the mesh with the minimum z-coordinate
    '''

    # Find the vertex of the mesh with the maximum x-coordinate
    max_x_idx = np.argmax(mesh.vertices[:, 0])
    max_x = mesh.vertices[max_x_idx]

    # Find the vertex of the mesh with the maximum y-coordinate
    max_y_idx = np.argmax(mesh.vertices[:, 1])
    max_y = mesh.vertices[max_y_idx]

    # Find the vertex of the mesh with the maximum z-coordinate
    max_z_idx = np.argmax(mesh.vertices[:, 2])
    max_z = mesh.vertices[max_z_idx]

    # Find the vertex of the mesh with the minimum x-coordinate
    min_x_idx = np.argmin(mesh.vertices[:, 0])
    min_x = mesh.vertices[min_x_idx]

    # Find the vertex of the mesh with the minimum y-coordinate
    min_y_idx = np.argmin(mesh.vertices[:, 1])
    min_y = mesh.vertices[min_y_idx]

    # Find the vertex of the mesh with the minimum z-coordinate
    min_z_idx = np.argmin(mesh.vertices[:, 2])
    min_z = mesh.vertices[min_z_idx]


    return [max_x, max_y, max_z, min_x, min_y, min_z]

def plane_equation_from_pos_dir(plane_pos, plane_dir):
    
    # Normalize the plane direction
    # normal_dir = plane_dir / np.linalg.norm(plane_dir)

    # Compute the plane equation parameters
    plane_params = np.concatenate((plane_dir, -np.array([np.dot(plane_pos, plane_dir)])))

    return plane_params

# Not working as intended
def is_point_inside_polytope(point, planes):
    for plane in planes:
        if np.dot(plane[:3], point) + plane[3] > 0:
            return False
    return True

def intersect_line_plane(p1, p2, plane_normal, plane_d):
    """
    Finds the intersection point of a line and a plane in 3D space.
    
    Parameters:
    - π1: np.array of shape (3,) representing the first point on the line.
    - π2: np.array of shape (3,) representing the second point on the line.
    - plane_normal: np.array of shape (3,) representing the normal vector of the plane.
    - plane_d: float representing the distance from the origin (d in plane equation).
    
    Returns:
    - intersection_point: np.array of shape (3,) representing the intersection point, or None if the line is parallel to the plane.
    """
    p1 = np.array([p1.x, p1.y, p1.z])
    p2 = np.array([p2.x, p2.y, p2.z])
    # Vector direction of the line
    line_direction = p2 - p1
    
    # Compute the denominator of the equation for t
    denominator = np.dot(plane_normal, line_direction)
    
    # If denominator is zero, the line is parallel to the plane
    if abs(denominator) < 1e-6:
        print("The line is parallel to the plane.")
        return None
    
    # Compute the numerator of the equation for t
    numerator = -(np.dot(plane_normal, p1) + plane_d)
    
    # Solve for t
    t = numerator / denominator
    
    # Compute the intersection point
    intersection_point = p1 + t * line_direction
    return intersection_point

def get_all_points_of_cuboid(cuboid: Cuboid3D, self=0):
    '''Get all the corner points of the cuboid.'''
    
    # Top max and bottom min points of the cuboid
    max_point = np.array([cuboid.x_max, cuboid.y_max, cuboid.z_max])
    min_point = np.array([cuboid.x_min, cuboid.y_min, cuboid.z_min])
    
    # Cuboid diamensions
    x_length = max_point[0] - min_point[0]
    z_length = max_point[2] - min_point[2]
    
    # Get all the corners of the cuboid
    bottom_corners = [Point3D([min_point[0], min_point[1], min_point[2]], color=Color.CYAN), 
                      Point3D([min_point[0] +x_length, min_point[1], min_point[2]], color=Color.GREY), 
                      Point3D([min_point[0] +x_length, min_point[1], min_point[2]+z_length], color=Color.ORANGE), 
                      Point3D([min_point[0], min_point[1], min_point[2]+z_length], color=Color.GREEN)]
    top_corners = [Point3D([max_point[0], max_point[1], max_point[2]], color=Color.RED), 
                   Point3D([max_point[0] - x_length, max_point[1], max_point[2]], color=Color.YELLOW), 
                   Point3D([max_point[0] -x_length, max_point[1], max_point[2]-z_length], color=Color.MAGENTA), 
                   Point3D([max_point[0], max_point[1], max_point[2]-z_length], color=Color.BLACK)]
    
    # Visualize the corners
    if self:
        for i in range(4):
            randomint = np.random.randint(0, 10000)
            self.addShape(bottom_corners[i], f"bottom_corners_{randomint}")
            self.addShape(top_corners[i], f"top_corners_{randomint}")
    
    # Combine the bottom and top corners
    corners = bottom_corners + top_corners
    return corners

def get_all_lines_of_cuboid(cuboid:Cuboid3D) -> LineSet3D:
    '''Returns all the lines of a cuboid.

    Args:
        cuboid: The Cuboid3D object

    Returns:
        lines: A list of all the lines of the cuboid
    '''
    lines = LineSet3D()
    # Get all the vertices of the cuboid
    corners = get_all_points_of_cuboid(cuboid)

    # Get all the lines of the cuboid
    lines.add(Line3D(corners[0], corners[1]))
    lines.add(Line3D(corners[1], corners[2]))
    lines.add(Line3D(corners[2], corners[3]))
    lines.add(Line3D(corners[3], corners[0]))
    lines.add(Line3D(corners[4], corners[5]))
    lines.add(Line3D(corners[5], corners[6]))
    lines.add(Line3D(corners[6], corners[7]))
    lines.add(Line3D(corners[7], corners[4]))
    lines.add(Line3D(corners[0], corners[6]))
    lines.add(Line3D(corners[1], corners[7]))
    lines.add(Line3D(corners[2], corners[4]))
    lines.add(Line3D(corners[3], corners[5]))

    return lines

