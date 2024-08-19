import numpy as np
import open3d as o3d
import numpy as np



# Contruct a default plane pointing in the upward y direction 
def default_plane():
    # Define vertices of the plane (two triangles)
    vertices = np.array([[-1.0,  0.0,  -1.0],  # Vertex 0
                        [ 1.0,  0.0,  -1.0],  # Vertex 1
                        [ 1.0,  0.0,   1.0],  # Vertex 2
                        [-1.0,  0.0,   1.0]]) # Vertex 3

    # Define faces of the plane (two triangles)
    faces = np.array([[0, 1, 2],  # Triangle 1 (vertices 0-1-2)
                    [0, 2, 3]]) # Triangle 2 (vertices 0-2-3)

    # Create Open3D TriangleMesh object
    plane_mesh = o3d.geometry.TriangleMesh()
    plane_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    plane_mesh.triangles = o3d.utility.Vector3iVector(faces)

    # print(f"----{ len(vertices[faces]) }")

    return plane_mesh

# Generate a plane with given position and orientation  
def generate_plane(direction, translation):
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
    plane = default_plane()

    # Initial center and direction of the plane
    center = np.array([0, 0, 0])
    dir = np.array([0, 1, 0])  # Default direction is along y-axis

    # Get vertices of the default plane
    vertices = np.asarray(plane.vertices)

    # Compute Euler angles from direction vector
    euler_angles = compute_euler_angles(direction)

    # Convert Euler angles to rotation matrix
    rotation_matrix = euler_angles_to_rotation_matrix(euler_angles)

    # Apply rigid transformation to vertices
    transformed_vertices = vertices @ rotation_matrix.T + translation 
    plane.vertices = o3d.utility.Vector3dVector(transformed_vertices)

    # Transform the center of the plane
    transformed_center = center @ rotation_matrix.T + translation

    # Transform the direction vector of the plane
    transformed_dir = dir @ rotation_matrix.T

    return plane, transformed_center, transformed_dir

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