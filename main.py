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
        self.meshes = {}
        self.plane_mesh = self.default_plane()
        self.addShape(self.plane_mesh, "plane")
        self.show_mesh()

    # Contruct a default plane pointing in the upward y direction 
    def default_plane(self, size:float = 2.0) -> Mesh3D:
        # Define vertices of the plane (two triangles)
        vertices = np.array([[-size,  0.0,  -size],  # Vertex 0
                            [ size,  0.0,  -size],   # Vertex 1
                            [ size,  0.0,   size],   # Vertex 2
                            [-size,  0.0,   size]])  # Vertex 3

        # Define triangles of the plane (two triangles)
        triangles = np.array([[0, 1, 2],        # Triangle 1 (vertices 0-1-2)
                            [0, 2, 3]])         # Triangle 2 (vertices 0-2-3)

        plane_mesh = Mesh3D(color=Color.GRAY)
        plane_mesh.vertices = vertices
        plane_mesh.triangles = triangles

        # Making a 2*size x 2*size grid on the plane
        # Vertical lines
        self.vertical_lines = LineSet3D()
        for i in range(1, 4):
            line = Line3D([vertices[0][0] + i, vertices[0][1], vertices[0][2]], 
                          [vertices[0][0] + i, vertices[0][1], vertices[0][2] + 2*size], 
                          color=Color.RED)
            self.vertical_lines.add(line)
            self.addShape(line, f"line{i}")


        # Horizontal lines
        self.horizontal_lines = LineSet3D()
        for i in range(1, 4):
            line = Line3D([vertices[0][0], vertices[0][1], vertices[0][2] + i], 
                          [vertices[0][0] + 2*size, vertices[0][1], vertices[0][2] + i], 
                          color=Color.RED)
            self.horizontal_lines.add(line)
            self.addShape(line, f"line{i+3}")
        
          
        return plane_mesh

    def show_mesh(self):
        self.mesh = Mesh3D("models/Helicopter.obj", color=Color.BLACK)
        self.randomize_mesh(self.mesh)

        self.mesh1 = Mesh3D("models/Helicopter.obj", color=Color.RED)   
        self.randomize_mesh(self.mesh1)

        self.mesh2 = Mesh3D("models/Helicopter.obj", color=Color.GREEN)
        self.randomize_mesh(self.mesh2)

        self.mesh3 = Mesh3D("models/Helicopter.obj", color=Color.BLUE)
        self.randomize_mesh(self.mesh3)

        self.mesh4 = Mesh3D("models/Helicopter.obj", color=Color.YELLOW)
        self.randomize_mesh(self.mesh4)

        for mesh in self.meshes.keys():
            print(mesh)
        
    def randomize_mesh(self, mesh: Mesh3D, trans_thresold:int = 2) -> Mesh3D:
        '''Randomizes the mesh and normalize inside the unit sphere.

        Args:
            mesh: The mesh

        Returns:
            mesh: The randomized mesh
        '''

        # Fit the mesh into the unit sphere
        mesh = self.unit_sphere_normalization(mesh)

        # Randomly translate the mesh
        translation_vector = np.array([random.randint(-trans_thresold, trans_thresold), 
                                       random.randint(-trans_thresold, trans_thresold), 
                                       random.randint(-trans_thresold, trans_thresold)])
        center = np.array([0, 0, 0])
        dir = np.array([1, 0, 0])
        vertices = mesh.vertices
        rotation_matrix = get_rotation_matrix(center, dir)
        transformed_vertices = vertices @ rotation_matrix.T + translation_vector
        mesh.vertices = transformed_vertices

        # Give the mesh a random name
        name = random.randint(0, 100)

        # Add the mesh to the scene
        self.addShape(mesh, f"drone{name}")

        # Store the mesh in the dictionary
        self.meshes[f"drone{name}"] = mesh

        return mesh

    def unit_sphere_normalization(self, mesh:Mesh3D) -> Mesh3D:
        '''Applies unit sphere normalization.

        Args:
            mesh: The mesh

        Returns:
            normalized_mesh: The normalized mesh
        '''

        mesh.vertices = np.array(mesh.vertices)
        max_distance = np.max(np.linalg.norm(mesh.vertices, axis=1))
        mesh.vertices /= max_distance

        return mesh
        



if __name__ == "__main__":
    scene = Project()
    scene.mainLoop()








