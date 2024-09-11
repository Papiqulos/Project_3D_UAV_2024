import numpy as np
import heapq
from vvrpywork.shapes import (
    Point3D, Sphere3D
)


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
