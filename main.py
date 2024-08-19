import open3d as o3d
import numpy as np
import random
from vvrpywork.constants import Key, Mouse, Color
from vvrpywork.scene import Scene3D, get_rotation_matrix, world_space
from vvrpywork.shapes import (
    Point3D, Line3D, Arrow3D, Sphere3D, Cuboid3D, Cuboid3DGeneralized,
    PointSet3D, LineSet3D, Mesh3D
)
import utility as U

WIDTH, HEIGHT = 800, 600


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
        self.N = 4

        # Create the landing pad
        self.landing_pad()

        # Testing
        # for mesh in self.meshes.keys():
        #     print(mesh)

        # for plane in self.planes.keys():
        #     print(plane)
    
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
                        self.get_aabb(mesh, mesh_name)
                           
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
    
    def landing_pad(self, size:float = 4.0) -> None:
        '''Construct an NxN landing pad.
        
        Args:
            size: The size of the landing pad
        '''

        # List of colours for the planes
        colours = [Color.GRAY, 
                   Color.WHITE]
        
        for i in range(self.N):
            for j in range(self.N):
                colour = colours[(i+j)%len(colours)]

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

        # List of colours for the drones
        colours = [Color.RED, 
                   Color.GREEN, 
                   Color.BLUE, 
                   Color.YELLOW, 
                   Color.BLACK, 
                   Color.WHITE, 
                   Color.GRAY, 
                   Color.CYAN, 
                   Color.MAGENTA, 
                   Color.ORANGE]

        for i in range(num_drones):
            colour = colours[i%len(colours)]
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

    def o3d_to_mesh(self, o3d_mesh:o3d.geometry.TriangleMesh) -> Mesh3D:
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
    
    def mesh_to_o3d(self, mesh:Mesh3D) -> o3d.geometry.TriangleMesh:
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

    def ch_o3d(self, mesh:Mesh3D) -> None:
        '''Construct the convex hull of the mesh using Open3D.'''

        # Convert the Mesh3D object to an Open3D mesh
        o3d_mesh = self.mesh_to_o3d(mesh)

        # Compute the convex hull using Open3D
        hull, _ = o3d_mesh.compute_convex_hull()

        # Convert the Open3D mesh to a Mesh3D object
        ch_mesh = self.o3d_to_mesh(hull)

        return ch_mesh

    def show_convex_hull(self, mesh:Mesh3D, mesh_name:str) -> None:
        '''Construct the convex hull of the mesh and put in the scene.

        Args:
            mesh: The mesh
            mesh_name: The name of the mesh
        '''
        # Compute the convex hull using Open3D
        ch_mesh = self.ch_o3d(mesh)

        # Add the convex hull to the scene
        self.addShape(ch_mesh, f"convex_hull_{mesh_name}")

        # Store the convex hull in the dictionary
        self.convex_hulls[f"convex_hull_{mesh_name}"] = ch_mesh

    def get_aabb(self, mesh: Mesh3D, mesh_name:str) -> None:
        '''Computes the axis-aligned bounding box (AABB) of a mesh.
        
        Args:
            mesh: The mesh
            mesh_name: The name of the mesh  
        '''
        
        vertices = np.array(mesh.vertices)
        
        # Compute the minimum and maximum coordinates along each axis
        min_coords = np.min(vertices, axis=0)
        max_coords = np.max(vertices, axis=0)
        
        # Create the AxisAlignedBoundingBox object
        aabb = Cuboid3D(p1=min_coords, p2=max_coords, color=Color.RED, filled=False)

        # Add the AABB to the scene
        self.addShape(aabb, f"aabb_{mesh_name}")

        # Store the AABB in the dictionary
        self.aabbs[f"aabb_{mesh_name}"] = aabb
    
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

        mesh_center = np.mean(mesh.vertices, axis=0)

        for i, direction in enumerate(directions):
            if i <= 6: 
                continue
            # Create a plane mesh with specified orientation and position
            plane, _, _ = U.generate_plane(direction, mesh_center)
            plane = self.o3d_to_mesh(plane)
            plane.color = Color.GREEN

            # Add the plane to the scene
            self.addShape(plane, f"kdop{i}_{mesh_name}")

            # Store the k-DOP in the dictionary
            self.kdops[f"kdop{i}_{mesh_name}"] = plane




if __name__ == "__main__":
    scene = Project()
    scene.mainLoop()








