import open3d as o3d
import numpy as np
import random
from vvrpywork.constants import Key, Mouse, Color
from vvrpywork.scene import Scene3D, get_rotation_matrix, world_space
from vvrpywork.shapes import (
    Point3D, Line3D, Arrow3D, Sphere3D, Cuboid3D, Cuboid3DGeneralized,
    PointSet3D, LineSet3D, Mesh3D
)

WIDTH, HEIGHT = 800, 600


class Project(Scene3D):

    def __init__(self):
        super().__init__(WIDTH, HEIGHT, "Project")

        # Dictionary to store the meshes
        self.meshes = {str: Mesh3D}

        # Dictionary to store the planes
        self.planes = {str: Cuboid3D}

        # Dimension of the landing pad
        self.N = 4
        self.landing_pad()
        self.show_drones()

        for mesh in self.meshes.keys():
            print(mesh)

        for plane in self.planes.keys():
            print(plane)

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
        self.addShape(mesh, f"drone{drone_id}")

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

if __name__ == "__main__":
    scene = Project()
    scene.mainLoop()








