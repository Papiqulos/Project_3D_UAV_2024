import numpy as np
import open3d as o3d
import numpy as np
from vvrpywork.shapes import (
    Point3D, Mesh3D
)

def default_plane(size:float=1.0) -> o3d.geometry.TriangleMesh:
    '''
    Contruct a default plane pointing in the upward y direction 
    
    Args:
    - size: Size of the plane
    
    Returns:
    - plane_mesh: Open3D TriangleMesh object representing the plane.
    '''
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

def rotation_matrix_from_axis_angle(axis:np.ndarray, angle:float) -> np.ndarray:
    '''
    Create a rotation matrix from an axis-angle representation.
    
    Args:
    - axis: np.array of shape (3,) representing the rotation axis.
    - angle: float representing the rotation angle in radians.
    
    Returns:
    - R: np.array of shape (3, 3) representing the rotation matrix.
    '''
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

# Not used
def euler_angles_to_rotation_matrix(euler_angles:np.ndarray) -> np.ndarray:
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

# Not used
def compute_euler_angles(direction:np.ndarray) -> np.ndarray:
    '''
    Compute Euler angles from a direction vector.
    
    Args:
    - direction: np.array of shape (3,) representing the direction vector.
    
    Returns:
    - euler_angles: np.array of shape (3,) representing the Euler angles [alpha, beta, gamma].
    '''
    x, y, z = direction
    
    # Yaw (alpha)
    alpha = np.arctan2(y, x)
    
    # Pitch (beta)
    beta = np.arctan2(z, np.sqrt(x**2 + y**2))
    
    # Roll (gamma) is typically 0 for aligning vectors
    gamma = 0.0
    
    return np.array([alpha, beta, gamma])

def generate_plane(direction:np.ndarray, translation:np.ndarray, size:float = 1.0) -> tuple[Mesh3D, np.ndarray, np.ndarray]:
    """
    Generate a plane mesh with specified orientation and position.

    Parameters:
    - direction: Direction vector [dx, dy, dz] of the plane.
    - translation: Translation vector [tx, ty, tz] for position.
    - size: Size of the plane.

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

def intersection_of_three_planes(plane1:np.ndarray, plane2:np.ndarray, plane3:np.ndarray) -> np.ndarray:
    '''
    Compute the intersection point of three planes in 3D space.
    
    Args:
    - plane1: np.array of shape (4,) representing the plane equation parameters [a, b, c, d].
    - plane2: np.array of shape (4,) representing the plane equation parameters [a, b, c, d].
    - plane3: np.array of shape (4,) representing the plane equation parameters [a, b, c, d].
    
    Returns:
    - intersection_point: np.array of shape (3,) representing the intersection point, or None if the planes do not intersect.
    '''
    # Extract coefficients from the plane equations
    A = np.array([plane1[:3], plane2[:3], plane3[:3]])
    b = np.array([-plane1[3], -plane2[3], -plane3[3]])
    
    # Check if the determinant of A is non-zero
    if np.linalg.det(A) == 0:
        return
    
    # Solve the system of equations
    intersection_point = np.linalg.solve(A, b)
    
    return intersection_point

def plane_equation_from_pos_dir(plane_pos:np.ndarray, plane_dir:float) -> np.ndarray:
    """
    Compute the plane equation parameters from a position and direction vector.
    
    Args:
    - plane_pos: np.array of shape (3,) representing a point on the plane.
    - plane_dir: np.array of shape (3,) representing the direction vector of the plane.
    
    Returns:
    - plane_params: np.array of shape (4,) representing the plane equation parameters [a, b, c, d].
    """
    # Normalize the plane direction
    # normal_dir = plane_dir / np.linalg.norm(plane_dir)

    # Compute the plane equation parameters
    plane_params = np.concatenate((plane_dir, -np.array([np.dot(plane_pos, plane_dir)])))

    return plane_params

def intersect_line_plane(p1:Point3D|np.ndarray, p2:Point3D|np.ndarray, plane_normal:np.ndarray, plane_d:float) -> np.ndarray:
    """
    Finds the intersection point of a line and a plane in 3D space.
    
    Args:
    - p1: np.array of shape (3,) or Point3D object representing the first point on the line.
    - p2: np.array of shape (3,) or Point3D object representing the second point on the line.
    - plane_normal: np.array of shape (3,) representing the normal vector of the plane.
    - plane_d: float representing the distance from the origin (d in plane equation).
    
    Returns:
    - intersection_point: np.array of shape (3,) representing the intersection point, or None if the line is parallel to the plane.
    """
    if isinstance(p1, Point3D):
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

def get_convex_hull_of_pcd(points:np.ndarray) -> Mesh3D:
    '''Creates a triangle mesh from a set of points by creating the convex hull of the points.'''
    # Create Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    hull, _ = pcd.compute_convex_hull()
    hull = o3d_to_mesh(hull)
    
    return hull

def get_random_rotation_matrix(deflection=1.0, randnums=None):
    """
    Creates a random rotation matrix.
    
    deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random
    rotation. Small deflection => small perturbation.
    randnums: 3 random numbers in the range [0, 1]. If `None`, they will be auto-generated.
    """
    # from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c
    
    if randnums is None:
        randnums = np.random.uniform(size=(3,))
        
    theta, phi, z = randnums
    
    theta = theta * 2.0*deflection*np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0*np.pi  # For direction of pole deflection.
    z = z * 2.0*deflection  # For magnitude of pole deflection.
    
    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.
    
    r = np.sqrt(z)
    Vx, Vy, Vz = V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
        )
    
    st = np.sin(theta)
    ct = np.cos(theta)
    
    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))
    
    # Construct the rotation matrix  ( V Transpose(V) - I ) R.
    
    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M




