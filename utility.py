import numpy as np
import open3d as o3d
import numpy as np
from vvrpywork.constants import Color
from vvrpywork.shapes import (
    Point3D, Mesh3D, Cuboid3D
)
from itertools import combinations
import trimesh
from kdnode import KdNode

# Not used
def rSubset(arr, r):
    # return list of all subsets of length r
    # to deal with duplicate subsets use set(list(combinations(arr, r)))
    return set(list(combinations(arr, r)))

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
        mesh.vertex_colors = np.array(o3d_mesh.vertex_colors)

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
    o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(mesh.vertex_colors)

    return o3d_mesh

def get_nearest_point(point:np.ndarray, mesh:Mesh3D, lst:bool = True) -> Point3D|np.ndarray:
        '''Get the nearest point to a given point on the mesh.

        Args:
            point
            mesh 
            lst: If True, return the nearest point as a list, else return as a Point3D object

        Returns:
            nearest_point
        '''
        # Construct the k-d tree from the mesh vertices
        kd_tree = KdNode.build_kd_node(np.array(mesh.vertices))

        # Convert the point to Point3D object
        point = Point3D(np.array(point), color=Color.BLACK, size=3)
        # self.addShape(point, f"point{random.randint(0, 1000)}")

        nearest_node = KdNode.nearestNeighbor(point, kd_tree)
        if lst:
            nearest_point = nearest_node.point
        else:
            nearest_point = Point3D(nearest_node.point, color=Color.CYAN, size=3)

        return nearest_point

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
    
    theta = theta * 2.0 * deflection * np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0 * np.pi  # For direction of pole deflection.
    z = z * 2.0 * deflection  # For magnitude of pole deflection.
    
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

def intersect_cuboids(cuboid_a, cuboid_b):
    """
    Computes the intersection cuboid formed by two colliding cuboids, if any.

    Parameters:
    - corners_a: A numpy array of shape (8, 3) representing the 8 corners of the first cuboid.
    - corners_b: A numpy array of shape (8, 3) representing the 8 corners of the second cuboid.

    Returns:
    - intersect_min: A numpy array representing the minimum coordinates of the intersection cuboid.
    - intersect_max: A numpy array representing the maximum coordinates of the intersection cuboid.
    """
    # Find min and max coordinates for both cuboids
    min_a, max_a = np.array([cuboid_a.x_min, cuboid_a.y_min, cuboid_a.z_min]), np.array([cuboid_a.x_max, cuboid_a.y_max, cuboid_a.z_max])
    min_b, max_b = np.array([cuboid_b.x_min, cuboid_b.y_min, cuboid_b.z_min]), np.array([cuboid_b.x_max, cuboid_b.y_max, cuboid_b.z_max])
    
    # Compute the min and max coordinates of the intersection cuboid
    intersect_min = np.maximum(min_a, min_b)  # Take the maximum of the minimums
    intersect_max = np.minimum(max_a, max_b)  # Take the minimum of the maximums

    # Check if there is an intersection (if the min coordinates are less than the max coordinates)
    if np.all(intersect_min <= intersect_max):
        return intersect_min, intersect_max
    else:
        # print("The cuboids do not intersect.")
        return None, None
    
def collision(mesh1:Mesh3D, mesh2:Mesh3D, return_points:bool = False) -> bool|tuple[bool, np.ndarray]:
    """Check if two meshes are in collision using trimesh library.
    
    Args:
    - mesh1: First mesh
    - mesh2: Second mesh
    - points_flag: If True, return the collision points
    
    Returns:
    - True if the meshes are in collision, False otherwise
    - points: List of collision points if points_flag is True
    """

    trimesh1 = trimesh.Trimesh(vertices=mesh1.vertices, faces=mesh1.triangles)
    trimesh2 = trimesh.Trimesh(vertices=mesh2.vertices, faces=mesh2.triangles)
    collision_manager = trimesh.collision.CollisionManager()
    collision_manager.add_object("trimesh1", trimesh1)
    collision_manager.add_object("trimesh2", trimesh2)

    if return_points:
        collision, point_objs = collision_manager.in_collision_internal(return_data=True)
        points = np.array([point_obj.point for point_obj in point_objs])
        return collision, points
    else:
        return collision_manager.in_collision_internal()
        
def shift_center_of_mass(mesh:Mesh3D, new_center:np.ndarray) -> Mesh3D:
    """
    Shift the center of mass of a mesh to a new position.

    Parameters:
    - mesh: The input mesh
    - new_center: The new center of mass

    Returns:
    - mesh: The mesh with the center of mass shifted
    """
    # Compute the current center of mass
    current_center = np.mean(mesh.vertices, axis=0)

    # Compute the translation vector
    translation = new_center - current_center

    # Shift the vertices
    mesh.vertices += translation

    return mesh

def unit_sphere_normalization(mesh:Mesh3D) -> Mesh3D:
    '''Applies unit sphere normalization to the mesh.

    Args:
        mesh: The mesh

    Returns:
        normalized_mesh: The normalized mesh
    '''

    mesh.vertices = np.array(mesh.vertices)
    center = np.mean(mesh.vertices, axis=0)
    mesh.vertices -= center
    max_distance = np.max(np.linalg.norm(mesh.vertices, axis=1))
    mesh.vertices /= max_distance

    return mesh

def get_surface_normal(mesh:Mesh3D, collision_point:np.ndarray) -> np.ndarray:
    '''Computes the surface normal at a collision point on a mesh.

    Args:
        mesh: The mesh
        collision_point: The collision point

    Returns:
        normal: The surface normal at the collision point
    '''

    nearest_vertex = get_nearest_point(collision_point, mesh)

    # Get the nearest vertex index
    nearest_vertex_index = np.where(np.all(mesh.vertices == nearest_vertex, axis=1))[0][0]

    # Get the faces that contain the nearest vertex
    faces = np.array(mesh.triangles)
    vertex_faces = np.where(np.any(faces == nearest_vertex_index, axis=1))[0]

    # Compute the average normal of the faces
    normals = np.array(mesh.vertex_normals)
    normal = np.mean(normals[vertex_faces], axis=0)

    return normal

def get_projection(mesh:Mesh3D, plane:str) -> Mesh3D:
    """Get the projection of a mesh on a plane.
    
    Args:
    - mesh: The mesh
    - plane: The plane of projection ("xy", "xz", "yz")

    Returns:
    - proj_mesh: The projected mesh
    """
    vertices = np.array(mesh.vertices)
    v = vertices.copy()

    if plane == "xy":
        v[:, 2] = 0
    elif plane == "xz":
        v[:, 1] = 0
    elif plane == "yz":
        v[:, 0] = 0

    proj_mesh = Mesh3D(color=mesh.color)
    proj_mesh.vertices = v
    proj_mesh.triangles = mesh.triangles

    return proj_mesh

def get_min_max_directions(mesh:Mesh3D, directions:np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get the minimum and maximum extents as well as the dot products of a mesh in certain directions.
    
    Args:
    - mesh: The mesh
    - directions: The directions
    
    Returns:
    - min_extents: The minimum extents
    - max_extents: The maximum extents
    - dot_products: The dot products
    """
    vertices = np.array(mesh.vertices)

    # Distances for each direction
    dot_products = np.dot(vertices, directions.T)

    # Get the maximum and minimum dot products
    max_extents = np.max(dot_products, axis=0)
    min_extents = np.min(dot_products, axis=0)

    return min_extents, max_extents, dot_products

        
    
        

    






