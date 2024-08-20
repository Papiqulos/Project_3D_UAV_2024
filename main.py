import open3d as o3d
import numpy as np
import random
from vvrpywork.constants import Key, Mouse, Color
from vvrpywork.scene import Scene3D, get_rotation_matrix, world_space
from vvrpywork.shapes import (
    Point3D, Line3D, Arrow3D, Sphere3D, Cuboid3D, Cuboid3DGeneralized,
    PointSet3D, LineSet3D, Mesh3D
)
import heapq
import utility as U

WIDTH, HEIGHT = 1800, 900

COLOURS = [Color.RED, 
                   Color.GREEN, 
                   Color.BLUE, 
                   Color.YELLOW, 
                   Color.BLACK, 
                   Color.WHITE, 
                   Color.GRAY, 
                   Color.CYAN, 
                   Color.MAGENTA, 
                   Color.ORANGE]

COLOURS_BW = [Color.BLACK, Color.WHITE]


class Project(Scene3D):

    def __init__(self):
        super().__init__(WIDTH, HEIGHT, "Project", output=True, n_sliders=1)
        self.printHelp()
        self.reset_sliders()

        # Dictionary to store the meshes
        self.meshes = {}

        # Dictionary to store the planes
        self.planes = {}

        # Dictionary to store the convex hulls
        self.convex_hulls = {}

        # Dictionary to store the axis-aligned bounding boxes (AABBs)
        self.aabbs = {}

        # Dictionary to store the k-discrete oriented polytopes (k-DOPs)
        self.kdops = {}

        # Dimension of the landing pad
        self.N = 5

        # Create the landing pad
        self.landing_pad(self.N)
    
    # List of commands
    def printHelp(self):
        self.print("\
        R: Clear scene\n\
        S: Show drones in random positions\n\
        C: Toggle convex hulls\n\
        A: Toggle AABBs\n\
        K: Toggle k-DOPs\n\n")

    def on_key_press(self, symbol, modifiers):

        if symbol == Key.R:
            self.clear_scene()

        if symbol == Key.S:
            if self.meshes:
                self.print("Drones already exist. Clear them first.")
                return
            self.show_drones(self.num_of_drones)
        
        if symbol == Key.C:

            if not self.meshes:
                self.print("No drones to show convex hulls for.")
                return
            
            if self.meshes:
                if self.convex_hulls:
                    for mesh_name, _ in self.meshes.items():
                        self.removeShape(f"convex_hull_{mesh_name}")
                    self.convex_hulls = {}
                else:
                    for mesh_name, mesh in self.meshes.items():
                        self.show_convex_hull(mesh, mesh_name)
        
        if symbol == Key.A:
            
            if not self.meshes:
                self.print("No drones to show AABBs for.")
                return

            if self.meshes:
                if self.aabbs:
                    for mesh_name, _ in self.meshes.items():
                        self.removeShape(f"aabb_{mesh_name}")
                    self.aabbs = {}
                else:
                    for mesh_name, mesh in self.meshes.items():
                        self.show_aabb(mesh, mesh_name)
                           
        if symbol == Key.K:
            if not self.meshes:
                self.print("No drones to show k-DOPs for.")
                return

            if self.meshes:
                if self.kdops:
                    for kdop_name, _ in self.kdops.items():
                        self.removeShape(kdop_name)
                    self.kdops = {}
                else:
                    for mesh_name, mesh in self.meshes.items():
                        self.get_kdop(mesh, mesh_name)
                           
    def reset_sliders(self):
        self.set_slider_value(0, 0.1)
    
    def on_slider_change(self, slider_id, value):

        if slider_id == 0:
            self.num_of_drones = int(10 * value)

    def clear_scene(self) -> None:
        '''Clear all the drones from the scene.'''
        for mesh_name, _ in self.meshes.items():
            self.removeShape(mesh_name)
            self.removeShape(f"convex_hull_{mesh_name}")
            self.removeShape(f"aabb_{mesh_name}")
        
        for kdop_name, _ in self.kdops.items():
            self.removeShape(kdop_name)
        self.meshes = {}
        self.convex_hulls = {}
        self.aabbs = {}
        self.kdops = {}
    
    def landing_pad(self, size:float) -> None:
        '''Construct an NxN landing pad.
        
        Args:
            size: The size of the landing pad
        '''
        
        for i in range(self.N):
            for j in range(self.N):
                colour = COLOURS_BW[(i+j)%len(COLOURS_BW)]

                plane = Cuboid3D(p1=[2*i - size, 0, 2*j - size], 
                                 p2=[2*i+2 - size, -0.2, 2*j+2 - size], 
                                 color=colour, 
                                 filled = True)
                
                plane_id = f"plane_{i}_{j}"
                self.addShape(plane, plane_id)
                self.planes[plane_id] = plane
        
    def show_drones(self, num_drones:int = 10) -> None:
        '''Show a certain number of drones in random positions.

        Args:
            num_drones: The number of drones
        '''

        for i in range(num_drones):
            colour = COLOURS[i%len(COLOURS)]
            mesh = Mesh3D(path="models/Helicopter.obj", color=colour)
            mesh = self.randomize_mesh(mesh, i)
            self.meshes[f"drone_{i}"] = mesh

    def randomize_mesh(self, mesh: Mesh3D, drone_id:int, trans_thresold:float = 2.0) -> Mesh3D:
        '''Fits the mesh into the unit sphere and randomly translates it.

        Args:
            mesh: The mesh
            drone_id: The ID of the drone
            trans_thresold: The translation threshold

        Returns:
            mesh: The randomized mesh
        '''

        # Fit the mesh into the unit sphere
        mesh = self.unit_sphere_normalization(mesh)

        # Randomly translate the mesh
        translation_vector = np.array([random.uniform(-trans_thresold, trans_thresold), 
                                       random.uniform(0.5, trans_thresold), 
                                       random.uniform(-trans_thresold, trans_thresold)])
        center = np.array([0, 0, 0])
        dir = np.array([1, 0, 0])
        vertices = mesh.vertices
        rotation_matrix = get_rotation_matrix(center, dir)
        transformed_vertices = vertices @ rotation_matrix.T + translation_vector
        mesh.vertices = transformed_vertices

        # Add the mesh to the scene
        self.addShape(mesh, f"drone_{drone_id}")

        return mesh

    def unit_sphere_normalization(self, mesh:Mesh3D) -> Mesh3D:
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

    def get_convex_hull(self, mesh:Mesh3D) -> Mesh3D:
        '''Construct the convex hull of the mesh using Open3D.'''

        # Convert the Mesh3D object to an Open3D mesh
        o3d_mesh = U.mesh_to_o3d(mesh)

        # Compute the convex hull using Open3D
        hull, _ = o3d_mesh.compute_convex_hull()

        # Convert the Open3D mesh to a Mesh3D object
        ch_mesh = U.o3d_to_mesh(hull)

        return ch_mesh

    def show_convex_hull(self, mesh:Mesh3D, mesh_name:str) -> None:
        '''Construct the convex hull of the mesh and put in the scene.

        Args:
            mesh: The mesh
            mesh_name: The name of the mesh
        '''
        # Compute the convex hull using Open3D
        ch_mesh = self.get_convex_hull(mesh)

        # Add the convex hull to the scene
        self.addShape(ch_mesh, f"convex_hull_{mesh_name}")

        # Store the convex hull in the dictionary
        self.convex_hulls[f"convex_hull_{mesh_name}"] = ch_mesh

    def get_aabb(self, mesh:Mesh3D, mesh_name:str) -> Cuboid3D:
        '''Computes the axis-aligned bounding box (AABB) of a mesh.'''

        vertices = np.array(mesh.vertices)
        
        # Compute the minimum and maximum coordinates along each axis
        min_coords = np.min(vertices, axis=0)
        max_coords = np.max(vertices, axis=0)
        
        # Create the AxisAlignedBoundingBox object
        aabb = Cuboid3D(p1=min_coords, p2=max_coords, color=Color.RED, filled=False)

        # Store the AABB in the dictionary
        self.aabbs[f"aabb_{mesh_name}"] = aabb

        return aabb

    def show_aabb(self, mesh: Mesh3D, mesh_name:str) -> None:
        '''Computes the axis-aligned bounding box (AABB) of a mesh and shows it in the scene.
        
        Args:
            mesh: The mesh
            mesh_name: The name of the mesh  
        '''
        
        # Create the AxisAlignedBoundingBox object
        aabb = self.get_aabb(mesh, mesh_name)

        # Add the AABB to the scene
        self.addShape(aabb, f"aabb_{mesh_name}")
    
    def get_nearest_point(self, point:list|np.ndarray, mesh:Mesh3D) -> Point3D:
        '''Get the nearest point to a given point on the mesh.

        Args:
            point: The point
            mesh: The mesh

        Returns:
            nearest_point: The nearest point
        '''
        # Construct the k-d tree from the mesh vertices
        kd_tree = KdNode.build_kd_node(np.array(mesh.vertices))

        # Convert the point to Point3D object
        point = Point3D(np.array(point), color=Color.BLACK, size=3)
        self.addShape(point, f"point{random.randint(0, 1000)}")

        nearest_node = KdNode.nearestNeighbor(point, kd_tree)
        nearest_point = Point3D(nearest_node.point, color=Color.CYAN, size=3)

        return nearest_point

    def get_all_cuboid_points(self, cuboid:Cuboid3D) -> np.ndarray:

        points = []
        
        return np.array(points)

    def get_kdop(self, mesh: Mesh3D, mesh_name:str) -> None:
        '''Computes the k-discrete oriented polytope (k-DOP) of a mesh.
        
        Args:
            mesh: The mesh
            mesh_name: The name of the mesh
        '''

        # 14 directions for the 14-DOP
        directions = np.array([
            [1, 0, 0], [-1, 0, 0],  # ±X-axis
            [0, 1, 0], [0, -1, 0],  # ±Y-axis
            [0, 0, 1], [0, 0, -1],  # ±Z-axis
            [1, 1, 0], [-1, -1, 0], [1, -1, 0], [-1, 1, 0],  # XY-plane diagonals
            [1, 0, 1], [-1, 0, -1], [1, 0, -1], [-1, 0, 1],  # XZ-plane diagonals
        ])

        if not self.aabbs:
            for mesh_name, mesh in self.meshes.items():
                self.get_aabb(mesh, mesh_name)
        for aabb in self.aabbs.values():

            # points = self.get_all_cuboid_points(aabb)
            # for point in points:
            #     near = self.get_nearest_point(point, mesh)
            #     self.addShape(near, f"nearest_point{random.randint(0, 1000)}")

            point1 = np.array([aabb.x_min, aabb.y_min, aabb.z_min])
            point2 = np.array([aabb.x_max, aabb.y_max, aabb.z_max])

            near1 = self.get_nearest_point(point1, mesh)
            near2 = self.get_nearest_point(point2, mesh)

            self.addShape(near1, f"nearest_point{random.randint(0, 1000)}")
            self.addShape(near2, f"nearest_point{random.randint(0, 1000)}")

            # Construct a plane whose normal is the vector from the point to the nearest point on the mesh
            near1 = np.array([near1.x, near1.y, near1.z])
            near2 = np.array([near2.x, near2.y, near2.z])

            normal1 = near1 - point1
            normal2 = near2 - point2

            plane1, _, _ = U.generate_plane([1,0,0], near1)

            plane1 = U.o3d_to_mesh(plane1)

            self.addShape(plane1, f"plane{random.randint(0, 1000)}")
            # self.addShape(plane2, f"plane{random.randint(0, 1000)}")



            


class KdNode:

    def __init__(self, point, left_child, right_child):
        """
        Initializes a KdNode object with the given point and child nodes.

        Args:
        - point: The point associated with the node.
        - left_child: The left child node.
        - right_child: The right child node.
        """
        self.point = point
        self.left_child = left_child
        self.right_child = right_child

    @staticmethod
    def build_kd_node(pts: np.array) -> 'KdNode':
        """
        Build a k-d tree from a set of points.

        Args:
        - pts (np.array): The set of points.

        Returns:
        - KdNode: The root node of the constructed k-d tree.
        """
        def _build_kd_node(pts: np.array, level: int) -> 'KdNode':
            """
            Recursively build a k-d tree from a set of points.

            Args:
            - pts (np.array): The set of points.
            - level (int): The current level in the k-d tree.

            Returns:
            - KdNode: The root node of the constructed k-d tree.
            """
            if len(pts) == 0:
                return None
            else:
                dim = 3
                axis = level % dim

                # Sort points based on the current axis.
                indices = np.argsort(pts[:, axis])
                sorted_pts = pts[indices]

                # Find the median point.
                median_idx = len(indices) // 2
                split_point = sorted_pts[median_idx]

                # Recursively build left and right subtrees.
                pts_left = sorted_pts[:median_idx]
                pts_right = sorted_pts[median_idx + 1:]
                
                left_child = _build_kd_node(pts_left, level + 1)
                right_child = _build_kd_node(pts_right, level + 1)

                return KdNode(split_point, left_child, right_child)

        # Start building the k-d tree from the root.
        root = _build_kd_node(pts, 0)
        return root

    @staticmethod
    def getNodesBelow(root: 'KdNode') -> np.ndarray:
        """
        Static method to collect all points below a given node in the k-d tree.

        Args:
        - root: The root node from which to start collecting points.

        Returns:
        - An ndarray containing all points below the given node.
        """
        def _getNodesBelow(node: 'KdNode', pts):
            """
            Recursive function to traverse the k-d tree and collect points below the given node in a depth-first manner 

            Args:
            - node: The current node being visited.
            - pts: List to collect points
            """
            # Visit the left child first if it exists.
            if node.left_child:
                _getNodesBelow(node.left_child, pts)

            # Then visit the right child if it exists.
            if node.right_child:
                _getNodesBelow(node.right_child, pts)

            # Finally, append the point of the current node to the list.
            pts.append(node.point)

            return

        # Initialize an empty list to collect points.
        pts = []

        # Recursively collect points below the root.
        _getNodesBelow(root, pts)

        # Convert the list of points to a numpy array and return.
        return np.array(pts)

    @staticmethod
    def getNodesAtDepth(root: 'KdNode', depth: int) -> list:
        """
        Collects nodes at the specified depth in a k-d tree.

        Args:
        - root: The root node of the k-d tree.
        - depth: The depth at which to collect nodes.

        Returns:
        - List of nodes at the specified depth.
        """
        def _getNodesAtDepth(node: 'KdNode', nodes: list, depth: int) -> None:
            """
            Recursive function to traverse the k-d tree and collect nodes at the specified depth.

            Args:
            - node: The current node being visited.
            - nodes: List to collect nodes at the specified depth.
            - depth: The depth of the target nodes with respect the current node being visited 
            """
            # Base case: If depth is 0, append the current node to the nodes list.
            if depth == 0:
                nodes.append(node)
            else:
                # If depth is greater than 0, recursively traverse the left and right children.
                if node.left_child:
                    _getNodesAtDepth(node.left_child, nodes, depth - 1)

                if node.right_child:
                    _getNodesAtDepth(node.right_child, nodes, depth - 1)

            return

        # Initialize an empty list to collect nodes.
        nodes = []

        # Recursively collect nodes at the specified depth starting from the root node.
        _getNodesAtDepth(root, nodes, depth)

        # Return the list of nodes collected at the specified depth.
        return nodes
    
    @staticmethod
    def inSphere(sphere: Sphere3D, root: 'KdNode'):
        """
        Find points within a sphere in a k-d tree.

        Args:
        - sphere (Sphere3D): The sphere to search within.
        - root (KdNode): The root node of the k-d tree.

        Returns:
        - np.ndarray: An array of points within the specified sphere.
        """
        def _inSphere(root, center, radius, level, pts):
            """
            Recursively search for points within a sphere in a k-d tree.

            Args:
            - root (KdNode): The current node being visited.
            - center (tuple): The center coordinates of the sphere.
            - radius (float): The radius of the sphere.
            - level (int): The current level in the k-d tree.
            - pts (list): List to collect points within the sphere.

            Returns:
            - np.ndarray: An array of points within the specified sphere.
            """
            if root is None:
                return
            axis = level % 3
            d_ = center[axis] - root.point[axis]
            is_on_left = d_ < 0

            # Recursively search the left subtree if the current node is on the left side of the sphere.
            if is_on_left:
                _inSphere(root.left_child, center, radius, level + 1, pts)

            # Recursively search the right subtree if the current node is on the right side of the sphere.
            else:
                _inSphere(root.right_child, center, radius, level + 1, pts)

            # Check if the current node is within the sphere.
            d = np.sum(np.square(center - root.point))

            if d <= radius ** 2:
                pts.append(root.point)

            # Recursively search the left subtree if the current node is on the right side of the sphere.
            if not is_on_left:
                _inSphere(root.left_child, center, radius, level + 1, pts)

            if is_on_left:
                _inSphere(root.right_child, center, radius, level + 1, pts)

            
            return pts

        # Extract center coordinates and radius from the input sphere.
        center = np.array([sphere.x, sphere.y, sphere.z])
        radius = sphere.radius 

        # Initialize an empty list to collect points within the sphere.
        pts = []

        # Recursively search for points within the sphere starting from the root of the k-d tree.
        _inSphere(root, center, radius, 0, pts)

        # Convert the list of points to a NumPy array and return it.
        return np.array(pts)

    @staticmethod
    def nearestNeighbor(test_pt: Point3D, root: 'KdNode') -> 'KdNode':  
        """
        Find the nearest neighbor of a given test point in the k-d tree.

        Args:
        - test_pt (Point3D): The test point.
        - root (KdNode): The root of the k-d tree.

        Returns:
        - KdNode: The nearest neighbor node in the k-d tree.
        """

        def _nearestNeighbor(root: 'KdNode', test_pt, level, dstar, pstar):
            """
            Recursively find the nearest neighbor of the test point in the k-d tree.

            Args:
            - root (KdNode): The current node being visited.
            - test_pt (np.ndarray): The coordinates of the test point.
            - level (int): The current level in the k-d tree.
            - dstar (float): The squared distance to the nearest neighbor found so far.
            - pstar (KdNode): The nearest neighbor node found so far.

            Returns:
            - Tuple[float, KdNode]: The updated squared distance to the nearest neighbor and the nearest neighbor node.
            """
            

            axis = level % 3
            d_ = test_pt[axis] - root.point[axis]

            is_on_left = d_ < 0 

            # Move to the appropriate subtree based on the relative position of the test point to the current node.
            if is_on_left:
                if root.left_child: 
                    dstar, pstar = _nearestNeighbor(root.left_child, test_pt, level + 1, dstar, pstar)
                if root.right_child and d_ ** 2 < dstar:  # Backtracking & pruning
                    dstar, pstar = _nearestNeighbor(root.right_child, test_pt, level + 1, dstar, pstar)
            else:
                if root.right_child: 
                    dstar, pstar = _nearestNeighbor(root.right_child, test_pt, level + 1, dstar, pstar)
                if root.left_child and d_ ** 2 < dstar:  # Backtracking & pruning
                    dstar, pstar = _nearestNeighbor(root.left_child, test_pt, level + 1, dstar, pstar)
            
            # Calculate the squared distance between the test point and the current node.
            d = np.sum(np.square(test_pt - root.point))

            # Update the nearest neighbor and its squared distance if needed.
            if d < dstar:
                dstar = d
                pstar = root

            return dstar, pstar
        
        # Initialize the variables for the nearest neighbor search.
        dstar = np.inf 
        pstar = None 

        # Convert the test point to a numpy array.
        test_pt = np.array([test_pt.x, test_pt.y, test_pt.z])

        # Start the nearest neighbor search from the root of the k-d tree.
        _, pstar = _nearestNeighbor(root, test_pt, 0, dstar, pstar)

        return pstar

    @staticmethod
    def nearestK(test_pt: Point3D, root: 'KdNode', K: int) -> list['KdNode']:
        """
        Find the K nearest neighbors of a given test point in the k-d tree.

        Args:
        - test_pt (Point3D): The test point.
        - root (KdNode): The root of the k-d tree.
        - K (int): The number of nearest neighbors to find.

        Returns:
        - List[KdNode]: A list of K nearest neighbor nodes in the k-d tree.
        """

        def _nearestK(root: 'KdNode', test_pt, K, level, heap, dstar):
            """
            Recursively find the K nearest neighbors of the test point in the k-d tree.

            Args:
            - root (KdNode): The current node being visited.
            - test_pt (np.array): The coordinates of the test point.
            - K (int): The number of nearest neighbors to find.
            - level (int): The current level in the k-d tree.
            - heap (list): A min-heap to store the K nearest neighbors found so far.
            - dstar (float): The squared distance to the farthest neighbor in the heap.

            Returns:
            - float: The updated squared distance to the farthest neighbor in the heap.
            """
            if root is None:
                return dstar

            axis = level % 3
            d_ = test_pt[axis] - root.point[axis]

            is_on_left = d_ < 0

            # Move to the appropriate subtree based on the relative position of the test point to the current node.
            if is_on_left:
                if root.left_child:
                    dstar = _nearestK(root.left_child, test_pt, K, level + 1, heap, dstar)
            else:
                if root.right_child:
                    dstar = _nearestK(root.right_child, test_pt, K, level + 1, heap, dstar)
            
            # Calculate the squared distance between the test point and the current node.
            d = np.sum(np.square(test_pt - root.point))

            # Update the heap with the current node if needed.
            if d < dstar:
                if len(heap) == K:
                    heapq.heappop(heap)
                heapq.heappush(heap, (-d, root))
                dstar = heap[0][0]
            
            # Check if the current node is within the hypersphere defined by the farthest neighbor in the heap.
            if len(heap) < K or d_ ** 2 < dstar:
                if is_on_left:
                    if root.right_child:
                        dstar = _nearestK(root.right_child, test_pt, K, level + 1, heap, dstar)
                else:
                    if root.left_child:
                        dstar = _nearestK(root.left_child, test_pt, K, level + 1, heap, dstar)

            # Add the current point to the heap if it is closer to the test point than the farthest neighbor in the heap.
            if len(heap) < K:
                heapq.heappush(heap, (-d, root))
            else:
                if d < -heap[0][0]:
                    heapq.heappop(heap)
                    heapq.heappush(heap, (-d, root))
            
            
            return dstar

        # Convert the test point to a numpy array.
        test_pt = np.array([test_pt.x, test_pt.y, test_pt.z])

        # Initialize variables for nearest neighbor search.
        dstar = np.inf
        heap = []

        # Start the nearest neighbor search from the root of the k-d tree.
        _nearestK(root, test_pt, K, 0, heap, dstar)
        
        # Retrieve the K nearest neighbor nodes from the heap.
        nodes = []
        while heap:
            _, node = heapq.heappop(heap)
            nodes.append(node)

        return nodes


        

        




if __name__ == "__main__":
    scene = Project()
    scene.mainLoop()








