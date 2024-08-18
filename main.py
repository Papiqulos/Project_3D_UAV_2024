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
        self.show_mesh()


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
        
        

    
    def randomize_mesh(self, mesh):
        translation_vector = np.array([random.randint(-10000, 1000), random.randint(-10000, 1000), random.randint(-10000, 1000)])

        center = np.array([0, 0, 0])
        dir = np.array([1, 0, 0])

        vertices = mesh.vertices
        rotation_matrix = get_rotation_matrix(center, dir)
        transformed_vertices = vertices @ rotation_matrix.T + translation_vector

        mesh.vertices = transformed_vertices
        name = random.randint(0, 100)

        self.addShape(mesh, f"drone{name}")
        self.meshes[f"drone{name}"] = mesh

        return mesh
        



if __name__ == "__main__":
    scene = Project()
    scene.mainLoop()








