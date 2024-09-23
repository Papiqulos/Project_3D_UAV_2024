import numpy as np
import random
import utility as U
import time
from vvrpywork.constants import Key, Color
from vvrpywork.scene import Scene3D, get_rotation_matrix
from vvrpywork.shapes import (
    Point3D, Cuboid3D,
    Mesh3D, Label3D, Line3D
)


# 14 directions for the 14-DOP
DIRECTIONS = np.array([
    [1, 0, 0],  # x-axis
    [0, 1, 0],  # y-axis
    [0, 0, 1],  # z-axis

    [-1, 0, 0], # -x-axis
    [0, -1, 0], # -y-axis
    [0, 0, -1], # -z-axis

    [1, 1, 1],  # diagonals    
    [-1, 1, 1], 
    [-1, 1, -1],
    [1, 1, -1],

    [-1, -1, -1],
    [1, -1, -1],
    [1, -1, 1],
    [-1, -1, 1]
    
])

WIDTH, HEIGHT = 1800, 900

COLOURS = [
            Color.RED, 
            Color.GREEN, 
            Color.BLUE, 
            Color.YELLOW, 
            Color.BLACK, 
            Color.WHITE, 
            Color.GRAY, 
            Color.CYAN, 
            Color.MAGENTA, 
            Color.ORANGE
          ]

DRONES = [
          "models/F52.obj", 
          "models/Helicopter.obj", 
          "models/quadcopter_scifi.obj",
          "models/v22_osprey.obj"
         ]

DRONES = DRONES[:2]

SPEEDS = np.array([
                  [0.0, 0.1, 0.0],
                  [0.0, -0.1, 0.0],
                  [0.0, 0.3, 0.3],
                  [0.1, 0.5, 0.1]
                  ])

SPEED_MAP = {DRONES[i]: SPEEDS[i] for i in range(len(DRONES))}

COLOURS_BW = [Color.BLACK, Color.WHITE]


class UavSim(Scene3D):

    def __init__(self):
        super().__init__(WIDTH, HEIGHT, "Project", output=True, n_sliders=1)

        # Move the camera to a better POV
        self.change_camera([0, 3, 7])
        # Print the help message
        self.display_commands()
        self.reset_sliders()

        ## Dictonaries to store the geometries
        # Dictionary to store the meshes
        self.meshes = {}

        # Dictionary to store the moving meshes
        self.moving_meshes = {}

        # Dictionary to store the landing pads
        self.landing_pads = {}

        # Dictionary to store the convex hulls
        self.convex_hulls = {}

        # Dictionary to store the axis-aligned bounding boxes (AABBs)
        self.aabbs = {}

        # Dictionary to store the k-discrete oriented polytopes (14-DOPs)
        self.kdops = {}

        # Dictionary to store the projections
        self.projections = {}

        # Dictionary to store the collision AABBs
        self.collision_aabbs = {}

        # Dictionary to store the collision points
        self.collision_points = {}

        # Dictionary to store all the misc geometries(For the labels and temporary geometries for testing)
        self.misc_geometries = {}

        ## Simulation variables
        self.dt = 0.03 # Time step
        self.paused = True # Pause the simulation
        self.paused_no_collisions = True # Pause the simulation without collisions
        self.pause_landing_simulation = True # Pause the landing protocol
        self.landed_meshes = {} # Drones that have landed
        self.last_update = time.time() # Time of the last update
        self.last_spawn_time = time.time() # Time of the last drone spawn
        self.spawn_interval = random.uniform(0.5, 1.5) # Time interval between drone spawns
        self.speeds = {} # Speeds of the drones

        ## Stat variables
        # Volumes' (Convex Hulls, AABBs, 14-DOPs) and Projections' Computation times
        self.ch_t = {"": 0}
        self.aabb_t = {"": 0}
        self.kdop_t = {"": 0}    
        self.projections_t = {"": 0}

        # Collision times
        self.ch_col_t = {"": 0}
        self.aabb_col_t = {"": 0}
        self.kdop_col_t = {"": 0}
        self.mesh_col_t = {"": 0}
        self.projections_col_t = {"": 0}
        
        ## Landing pad variables
        # Dimension of the landing pad
        self.N = 4

        # Create the landing pad
        self.landing_pad(self.N)
        self.y_bound = 10

        ## Bounding cuboid
        # Get the bounds of the landing pad
        top_left_corner_plane = self.landing_pads["plane_0_0"]
        bottom_right_corner_plane = self.landing_pads[f"plane_{self.N-1}_{self.N-1}"]

        top_left_point = top_left_corner_plane.get_all_points(lst=True)[6]
        
        bottom_right_point = bottom_right_corner_plane.get_all_points(lst=True)[4]
        
        self.bounding_cuboid = Cuboid3D(p1=top_left_point, p2=bottom_right_point+[0, self.y_bound, 0], color=Color.RED, filled=False)
        self.shown_bounds = True
    
    # INTERACTION FUNCTIONS
    def display_commands(self):
        self.print("\
        R: Clear scene\n\
        B: Toggle bounds\n\
        S: Show drones in random positions\n\
        P: Show drones in random positions and orientations\n\
        C: Toggle convex hulls\n\
        A: Toggle AABBs\n\
        K: Toggle k-DOPs\n\
        I: Toggle Projections(xy, xz, yz)\n\
        N: Check Collisions(AABBs)\n\
        L: Check Collisions(Convex Hulls)\n\
        M: Check Collisions(14-DOPs)\n\
        V: Check Collisions and Show Collision Points(Mesh3Ds)\n\
        O: Check Collisions(Projections)\n\
        T: Simulate\n\
        F: Simulate without collisions\n\
        Q: Show statistics\n\
        SPACE: Landing Protocol\n\
        \n\n")

    def on_key_press(self, symbol, modifiers):

        if symbol == Key.R:
            self.reset_scene()

        if symbol == Key.S:

            # if no drones exist, show them
            if self.meshes:
                self.print("Drones already exist. Clear them first.")
                return
            self.show_drones(self.num_of_drones, rand_rot=False, label=False)



            ## TESTING
            # mesh1 = Mesh3D(path="models/F52.obj", color=Color.RED)
            # mesh1 = U.unit_sphere_normalization(mesh1)
            # mesh2 = Mesh3D(path="models/Helicopter.obj", color=Color.GREEN)
            # mesh2 = U.unit_sphere_normalization(mesh2)

            # self.meshes["drone_1"] = mesh1
            # self.meshes["drone_2"] = mesh2

            # self.speeds["drone_1"] = SPEED_MAP[mesh1.path]
            # self.speeds["drone_2"] = SPEED_MAP[mesh2.path]

            # self.addShape(mesh1, "drone_1")
            # self.addShape(mesh2, "drone_2")

            # self.moving_meshes = self.meshes.copy()

            # self.move_drone_to_point(mesh1, "drone_1", [0, 2, 1])
            # self.move_drone_to_point(mesh2, "drone_2", [0, 8, 0])
            # self.meshes = self.moving_meshes.copy()
        
        if symbol == Key.P:

            # if no drones exist, show them
            if self.meshes:
                self.print("Drones already exist. Clear them first.")
                return
            self.show_drones(self.num_of_drones, rand_rot=True, label=False)

        if symbol == Key.C:

            # if no drones exist, show a message
            if not self.meshes:
                self.print("No drones to show convex hulls for.")
                return
            
            # if drones exist show/hide the convex hulls
            if self.meshes:

                # if convex hulls exist, remove them
                if self.convex_hulls:
                    for mesh_name in self.meshes.keys():
                        self.removeShape(f"convex_hull_{mesh_name}")
                    self.convex_hulls = {}
                # if convex hulls do not exist, show them
                else:
                    for mesh_name, mesh in self.meshes.items():
                        self.show_convex_hull(mesh, mesh_name)
        
        if symbol == Key.A:
            
            # if no drones exist, show a message
            if not self.meshes:
                self.print("No drones to show AABBs for.")
                return
            
            # if drones exist show/hide the AABBs
            if self.meshes:

                # if AABBs exist, remove them
                if self.aabbs:
                    for mesh_name, _ in self.meshes.items():
                        self.removeShape(f"aabb_{mesh_name}")
                    self.aabbs = {}
                # if AABBs do not exist, show them
                else:
                    for mesh_name, mesh in self.meshes.items():
                        self.show_aabb(mesh, mesh_name)
                           
        if symbol == Key.K:

            # if no drones exist, show a message
            if not self.meshes:
                self.print("No drones to show k-DOPs for.")
                return

            # if drones exist show/hide the 14-DOPs
            if self.meshes:
                if self.kdops:
                    for kdop_name in self.kdops.keys():
                        self.removeShape(kdop_name)
                    self.kdops = {}
                # if 14-DOPs do not exist, show them
                else:
                    for mesh_name, mesh in self.meshes.items():
                        self.show_14dop(mesh, mesh_name)
        
        if symbol == Key.I:
            
            # if no drones exist, show a message
            if not self.meshes:
                self.print("No drones to show Projections for.")
                return
            
            # if drones exist show/hide the projections
            if self.meshes:
                # if the projections exist, remove them
                if self.projections:
                    for mesh_name in self.meshes.keys():
                        self.removeShape(f"xy_back_{mesh_name}")
                        self.removeShape(f"xz_top_{mesh_name}")
                        self.removeShape(f"yz_right_{mesh_name}")
                    self.projections = {}
                # if projections do not exist, show them
                else:
                    for mesh_name, mesh in self.meshes.items():
                        self.show_projections(mesh, mesh_name)
        
        if symbol == Key.N:

            # if no drones exist, show a message
            if not self.meshes:
                self.print("No drones to show collisions for.")
                return

            # if drones exist, check for collisions using AABBs
            if self.meshes:

                # if collision AABBs exist, remove them and their labels
                if self.collision_aabbs:

                    for collision_mesh_name in self.collision_aabbs.keys():
                        self.removeShape(collision_mesh_name)
                    self.collision_aabbs = {}

                    # for label_name in self.misc_geometries.keys(): 
                    #     if "inter_cuboid" in label_name:
                    #         self.removeShape(label_name)

                # if collision AABBs do not exist, check for collisions
                else:
                    # Iterate over all the drones and check for collisions
                    for mesh_name, mesh in self.meshes.items():
                        for mesh_name2, mesh2 in self.meshes.items():
                            # Skip the same drone
                            if mesh_name > mesh_name2:
                                if self.collision_detection_aabbs(mesh, mesh_name, mesh2, mesh_name2):
                                    self.print(f"-AABB collision between {mesh_name} and {mesh_name2}")
 
        if symbol == Key.L:

            # if no drones exist, show a message
            if not self.meshes:
                self.print("No drones to show collisions for.")
                return
            
            # if drones exist, check for collisions using Convex Hulls
            if self.meshes:
                # Iterate over all the drones and check for collisions
                for mesh_name, mesh in self.meshes.items():
                    for mesh_name2, mesh2 in self.meshes.items():
                        # Skip the same drone
                        if mesh_name > mesh_name2:
                            if self.collision_detection_chs(mesh, mesh_name, mesh2, mesh_name2):
                                self.print(f"-Convex Hull collision between {mesh_name} and {mesh_name2}")
        
        if symbol == Key.M:

            # if no drones exist, show a message
            if not self.meshes:
                self.print("No drones to show collisions for.")
                return

            # if drones exist, check for collisions using 14-DOPs
            if self.meshes:

                # Iterate over all the drones and check for collisions
                for mesh_name, mesh in self.meshes.items():
                    for mesh_name2, mesh2 in self.meshes.items():
                        # Skip the same drone
                        if mesh_name > mesh_name2:
                            if self.collision_detection_kdops(mesh, mesh_name, mesh2, mesh_name2):
                                self.print(f"-14-DOP collision between {mesh_name} and {mesh_name2}")

        if symbol == Key.V:

            # if no drones exist, show a message
            if not self.meshes:
                self.print("No drones to show collisions for.")
                return
            # if drones exist, check for collisions using the meshes themselves
            if self.meshes:
                # If any collision points exist, remove them
                if self.collision_points:
                    for collision_point_name in self.collision_points.keys():
                        self.removeShape(collision_point_name)
                    self.collision_points = {}
                # Iterate over all the drones and check for collisions
                else:
                    for mesh_name, mesh in self.meshes.items():
                        for mesh_name2, mesh2 in self.meshes.items():
                            # Skip the same drone
                            if mesh_name > mesh_name2:
                                if self.collision_detection_meshes(mesh, mesh_name, mesh2, mesh_name2):
                                    self.print(f"-Mesh3D collision between {mesh_name} and {mesh_name2}")   
        
        if symbol == Key.O:

            # if no drones exist, show a message
            if not self.meshes:
                self.print("No drones to show collisions for.")
                return
            
            # if drones exist, check for collisions using the Projections
            if self.meshes:
                # Iterate over all the drones and check for collisions
                for mesh_name, mesh in self.meshes.items():
                    for mesh_name2, mesh2 in self.meshes.items():
                        # Skip the same drone
                        if mesh_name > mesh_name2:
                            if self.collision_detection_projections(mesh, mesh_name, mesh2, mesh_name2):
                                self.print(f"-Projections collision between {mesh_name} and {mesh_name2}")

        if symbol == Key.T:
            # Start/Pause the simulation
            self.paused = not self.paused
            self.print(f"-Paused: {self.paused}")

        if symbol == Key.F:
            # Start/Pause the simulation without collisions
            self.paused_no_collisions = not self.paused_no_collisions       
            self.print(f"-Paused_no_collision: {self.paused_no_collisions}")

        if symbol == Key.SPACE:
            # Start/Pause the landing protocol
            self.pause_landing_simulation = not self.pause_landing_simulation

        if symbol == Key.B:
            # Show the bounding cuboid
            if self.shown_bounds:
                self.addShape(self.bounding_cuboid, "bounding_cuboid")
                self.shown_bounds = not self.shown_bounds
            else:
                self.removeShape("bounding_cuboid")
                self.shown_bounds = not self.shown_bounds
        
        if symbol == Key.Q:
            self.stats()

    def reset_sliders(self):
        self.set_slider_value(0, 0.6)
    
    def on_slider_change(self, slider_id, value):

        if slider_id == 0:
            self.num_of_drones = int(10 * value)

    def reset_scene(self) -> None:
        '''REMOVE FROM THE SCENE AND DELETE FROM MEMORY EVERYTHING'''
        # AABBs, CHs, KDOPs, collision points, labels and misc geometries
        self.clear_attributes()
        # Drones and projections
        for mesh_name in self.meshes.keys():
            self.removeShape(mesh_name)
            self.removeShape(f"xy_back_{mesh_name}")
            self.removeShape(f"xy_front_{mesh_name}")
            self.removeShape(f"xz_top_{mesh_name}")
            self.removeShape(f"xz_bottom_{mesh_name}")
            self.removeShape(f"yz_right_{mesh_name}")
            self.removeShape(f"yz_left_{mesh_name}")
        # Intersecting cuboids
        for name in self.collision_aabbs.keys():
            self.removeShape(name)
        # Collision points
        for name in self.collision_points.keys():
            self.removeShape(name)
        for name in self.misc_geometries.keys():
            self.removeShape(name)
       
        self.meshes = {}
        self.aabbs = {}
        self.convex_hulls = {}
        self.kdops = {}
        self.projections = {}
        self.collision_aabbs = {}
        self.collision_points = {}
        self.misc_geometries = {}
        self.moving_meshes = {}
    
    def clear_attributes(self) -> None:
        # Remove any aabbs, chs, kdops, collision points and labels from the scene
        for mesh_name in self.meshes.keys():
            self.removeShape(f"aabb_{mesh_name}")
            self.removeShape(f"convex_hull_{mesh_name}")
            self.removeShape(f"14dop_{mesh_name}")
            self.removeShape(f"label_{mesh_name}")
        for name in self.collision_aabbs.keys():
            self.removeShape(name)
        for name in self.collision_points.keys():
            self.removeShape(name)
        for name in self.misc_geometries.keys():
            self.removeShape(name)
        
    # SETTING UP THE SCENE
    def landing_pad(self, size:float, height:float = 0.2) -> None:
        '''Construct an NxN landing pad.
        
        Args:
            size : The size of the landing pad
        '''
        
        for i in range(self.N):
            for j in range(self.N):
                colour = COLOURS_BW[(i+j)%len(COLOURS_BW)]

                plane = Cuboid3D(p1=[2*i - size, 0, 2*j - size], 
                                 p2=[2*i+2 - size, -height, 2*j+2 - size], 
                                 color=colour, 
                                 filled = True)
                
                plane_id = f"plane_{i}_{j}"
                self.addShape(plane, plane_id)
                self.landing_pads[plane_id] = plane
        
    def show_drones(self, num_drones:int = 10, rand_rot:bool = True, singular:bool = False, label:bool = False) -> None:
        '''Show a certain number of drones in random positions.

        Args:
            num_drones : The number of drones
            rand_rot : Whether to randomly rotate the drones
            singular : Whether to add one drone or multiple drones                
            label : Whether to add a label to the drones
        '''
        if num_drones > self.N**2:
            num_drones = self.N**2

        if singular:
            colour = COLOURS[random.randint(0, len(COLOURS)-1)]
            drone_path = DRONES[random.randint(0, len(DRONES)-1)]

            num_drones_in_scene = len(self.meshes.keys())
            id = num_drones_in_scene

            mesh = Mesh3D(path=drone_path, color=colour)
            mesh = self.randomize_mesh(mesh, id, label=label, rand_rot=rand_rot)
            self.meshes[f"drone_{id}"] = mesh
            self.speeds[f"drone_{id}"] = SPEED_MAP[mesh.path]
        else:   
            for i in range(num_drones):
                colour = COLOURS[i%len(COLOURS)]
                drone_path = DRONES[i%len(DRONES)]
                id = i
                mesh = Mesh3D(path=drone_path, color=colour)
                mesh = self.randomize_mesh(mesh, id, label=label, rand_rot=rand_rot)
                self.meshes[f"drone_{id}"] = mesh
                self.speeds[f"drone_{id}"] = SPEED_MAP[mesh.path]

        # Create a copy of the meshes to avoid modifying the original meshes
        self.moving_meshes = self.meshes.copy()

    def randomize_mesh(self, mesh: Mesh3D, drone_id:int, trans_thresold:float = 2.0, label:bool = False, rand_rot:bool = True) -> Mesh3D:
        '''Fits the mesh into the unit sphere and randomly translates it and rotates it.

        Args:
            mesh : The mesh
            drone_id : The ID of the drone
            trans_thresold : The translation threshold
            label : Whether to add a label to the drone
            rand_rot : Whether to randomly rotate the drone

        Returns:
            mesh : The randomized mesh
        '''

        # Fit the mesh into the unit sphere
        mesh = U.unit_sphere_normalization(mesh)
        vertices = mesh.vertices

        # Randomly translate the mesh
        translation_vector = np.array([random.uniform(-trans_thresold, trans_thresold), 
                                       random.uniform(0.5, self.y_bound), 
                                       random.uniform(-trans_thresold, trans_thresold)])
        
        if rand_rot:
            # Randomly rotate the mesh
            rotation_matrix = U.get_random_rotation_matrix()
        else:
            # No rotation
            center = np.array([0, 0, 0])
            dir = np.array([0, 1, 0])
            rotation_matrix = get_rotation_matrix(center, dir)

        if mesh.path == "models/Helicopter.obj":
            # Rotate the helicopter to face forward
            rotation_matrix = U.euler_angles_to_rotation_matrix([0, np.pi/2, 0])

        # Apply the translation and rotation to the vertices
        transformed_vertices = vertices @ rotation_matrix.T + translation_vector
        mesh.vertices = transformed_vertices

        # Add the mesh to the scene
        if label:
            label = Label3D(translation_vector, f"drone_{drone_id}", color=Color.BLACK)
            self.addShape(label, f"label_drone_{drone_id}")
            self.misc_geometries[f"label_drone_{drone_id}"] = label
        self.addShape(mesh, f"drone_{drone_id}")

        return mesh

    # VOLUMES' FUNCTIONS(Convex Hulls, AABBs, 14-DOPs) and Projections
    def get_convex_hull(self, mesh:Mesh3D, mesh_name:str) -> Mesh3D:
        '''Construct the convex hull of the mesh using Open3D.'''

        # Check if the convex hull already exists
        if f"convex_hull_{mesh_name}" in self.convex_hulls:
            return self.convex_hulls[f"convex_hull_{mesh_name}"]
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
        start = time.time()
        ch_mesh = self.get_convex_hull(mesh, mesh_name)
        end = time.time()
        self.ch_t[mesh_name] = end - start

        # Add the convex hull to the scene
        self.addShape(ch_mesh, f"convex_hull_{mesh_name}")

        # Store the convex hull in the dictionary
        self.convex_hulls[f"convex_hull_{mesh_name}"] = ch_mesh

    def get_aabb(self, mesh:Mesh3D, mesh_name:str) -> Cuboid3D:
        '''Computes the axis-aligned bounding box (AABB) of a mesh.'''

        # Check if the AABB already exists
        if f"aabb_{mesh_name}" in self.aabbs:
            return self.aabbs[f"aabb_{mesh_name}"]
        
        vertices = np.array(mesh.vertices)
        
        # Compute the minimum and maximum coordinates along each axis
        min_coords = np.min(vertices, axis=0)
        max_coords = np.max(vertices, axis=0)
        
        # Create the AxisAlignedBoundingBox object
        aabb = Cuboid3D(p1=min_coords, p2=max_coords, color=Color.RED, filled=False)

        # Store the AABB in the dictionary
        self.aabbs[f"aabb_{mesh_name}"] = aabb

        return aabb

    def show_aabb(self, mesh:Mesh3D, mesh_name:str) -> None:
        '''Computes the axis-aligned bounding box (AABB) of a mesh and shows it in the scene.
        
        Args:
            mesh: The mesh
            mesh_name: The name of the mesh  
        '''
        start = time.time()

        # Create the AxisAlignedBoundingBox object if it does not exist
        aabb = self.get_aabb(mesh, mesh_name)

        end = time.time()
        self.aabb_t[mesh_name] = end - start

        # Add the AABB to the scene
        self.addShape(aabb, f"aabb_{mesh_name}")
    
    def get_14dop(self, mesh:Mesh3D, mesh_name:str) -> Mesh3D:
        '''Computes the 14-discrete oriented polytope (k-DOP) of a mesh.
        
        Args:
            mesh: The mesh
            mesh_name: The name of the mesh
        
        Returns:
            ch_mesh: The 14-DOP
        '''
        
        # Check if the 14-DOP already exists
        if f"14dop_{mesh_name}" in self.kdops:
            return self.kdops[f"14dop_{mesh_name}"]
    
        vertices = np.array(mesh.vertices)
        
        # Get the dot products
        _, _, dot_products = U.get_min_max_directions(mesh, DIRECTIONS)
        
        # Get the indices of the maximum and minimum dot products
        max_indices = np.argmax(dot_products, axis=0)
        min_indices = np.argmin(dot_products, axis=0)

        # Get the vertices with the maximum and minimum dot products
        max_vertices = vertices[max_indices]
        max_vertices_faces = max_vertices[:6]
        max_vertices_corners = max_vertices[6:]

        min_vertices = vertices[min_indices]
        min_vertices_faces = min_vertices[:6]
        min_vertices_corners = min_vertices[6:]

        # Split the directions into faces and corners
        faces = DIRECTIONS[:6]
        corners = DIRECTIONS[6:]

        # Map the directions to the corresponding max and min vertices for the corners and faces
        faces_map = {tuple(dir): (max_vertices_faces[i], min_vertices_faces[i]) for i, dir in enumerate(faces)}
        corners_map = {tuple(dir): (max_vertices_corners[i], min_vertices_corners[i]) for i, dir in enumerate(corners)}
        
        # Dictionary to store the faces and corners
        faces_dict = {}
        corners_dict = {}

        # List to store the intersection points
        intersection_points = []

        # Get the plane equations for the faces and corners
        for i, (dir, vertex) in enumerate(faces_map.items()):
            plane, plane_dir, plane_pos = U.generate_plane(list(dir), vertex[0], 3)
            plane_params = U.plane_equation_from_pos_dir(plane_pos, plane_dir)
            faces_dict[plane] = plane_params

            # Visualize the plane
            # self.addShape(plane, f"face_plane_max_{i}")

            plane, plane_dir, plane_pos = U.generate_plane(list(dir), vertex[1], 3)
            plane_params = U.plane_equation_from_pos_dir(plane_pos, plane_dir)
            faces_dict[plane] = plane_params

            # Visualize the plane
            # self.addShape(plane, f"face_plane_min_{i}")

        for i, (dir, vertex) in enumerate(corners_map.items()):
            
            plane, plane_dir, plane_pos = U.generate_plane(list(dir), vertex[0], 3)
            plane_params = U.plane_equation_from_pos_dir(vertex[0], list(dir))
            corners_dict[plane] = plane_params

            # Visualize the plane
            # self.addShape(plane, f"corner_plane_max_{i}")

            plane, plane_dir, plane_pos = U.generate_plane(list(dir), vertex[1], 3)
            plane_params = U.plane_equation_from_pos_dir(vertex[1], list(dir))
            corners_dict[plane] = plane_params

            # Visualize the plane
            # self.addShape(plane, f"corner_plane_min_{i}")

        aabb = self.get_aabb(mesh, mesh_name)
        points_of_aabb = aabb.get_all_points()
        lines_of_aabb = aabb.get_all_lines()
        
        # Get the intersection points of the corner planes with the AABB
        for i, (plane, params) in enumerate(corners_dict.items()):
            # if i == 15:
                # self.addShape(plane, f"corner_plane_max_{i}")
                for j, line in enumerate(lines_of_aabb.lines):
                    # if j == 7:
                        p1 = points_of_aabb[line[0]]
                        p2 = points_of_aabb[line[1]]

                        
                        plane_normal = np.array(params[:3])
                        plane_d = params[3]

                        intersection_point = U.intersect_line_plane(p1, p2, plane_normal, plane_d)
                        if intersection_point is not None:
                            if aabb.check_point_in_cuboid(intersection_point):
                                intersection_points.append(intersection_point)
                                

        # Visualize the intersection points
        # print(f"inter points found : {len(intersection_points)}")

        # Make a convex hull of the intersection points
        intersection_points = np.array(intersection_points)
        ch_mesh = U.get_convex_hull_of_pcd(intersection_points)

        # Store the 14-DOP in the dictionary
        self.kdops[f"14dop_{mesh_name}"] = ch_mesh
        
        
        return ch_mesh

    def show_14dop(self, mesh:Mesh3D, mesh_name:str) -> None:
        '''Computes the 14-discrete oriented polytope (k-DOP) of a mesh and shows it in the scene.
        
        Args:
            mesh: The mesh
            mesh_name: The name of the mesh  
        '''
        start = time.time()
        kdop = self.get_14dop(mesh, mesh_name)
        end = time.time()
        self.kdop_t[mesh_name] = end - start

        # Add the 14-DOP to the scene
        self.addShape(kdop, f"14dop_{mesh_name}")

    def get_projections(self, mesh:Mesh3D, mesh_name:str) -> tuple[Mesh3D, Mesh3D, Mesh3D]:
        """Get the projections of the mesh to the faces of the bounding cuboid.
        
        Args:
            - mesh : The mesh
            - mesh_name : The name of the mesh
            
        Returns:
            mesh_xy_back : The projection of the mesh to the xy-plane of the back face of the bounding cuboid
            mesh_xz_top : The projection of the mesh to the xz-plane of the top face of the bounding cuboid
            mesh_yz_right : The projection of the mesh to the yz-plane of the right face of the bounding cuboid"""

        # Check if the projections already exist
        if f"xy_back_{mesh_name}" in self.projections and f"xz_top_{mesh_name}" in self.projections and f"yz_right_{mesh_name}" in self.projections:
            return self.projections[f"xy_back_{mesh_name}"], self.projections[f"xz_top_{mesh_name}"], self.projections[f"yz_right_{mesh_name}"]
        
        # Project the mesh to the xy-plane
        mesh_xy = U.get_projection(mesh, "xy")

        # Project the mesh to the xz-plane
        mesh_xz = U.get_projection(mesh, "xz")

        # Project the mesh to the yz-plane
        mesh_yz = U.get_projection(mesh, "yz")

        ## Move the projections to faces of the bounding cuboid
        # Get the centers of the faces of the bounding cuboid
        center = mesh.get_center(lst=True)
        tr_back, _, tr_top, _, tr_right, _ = self.bounding_cuboid.get_face_centers()

        tr_back = np.array([center[0], center[1], tr_back[2]])
        tr_top = np.array([center[0], tr_top[1], center[2]])
        tr_right = np.array([tr_right[0], center[1], center[2]])
        
        # Back  projection
        mesh_xy_back = U.shift_center_of_mass(mesh_xy, tr_back)
        # Add the projection to the dictionary
        self.projections[f"xy_back_{mesh_name}"] = mesh_xy_back
        
        # Top projection
        mesh_xz_top = U.shift_center_of_mass(mesh_xz, tr_top)
        # Add the projection to the dictionary
        self.projections[f"xz_top_{mesh_name}"] = mesh_xz_top
        
        # Right projection
        mesh_yz_right = U.shift_center_of_mass(mesh_yz, tr_right)
        self.projections[f"yz_right_{mesh_name}"] = mesh_yz_right
        

        return mesh_xy_back, mesh_xz_top, mesh_yz_right

    def show_projections(self, mesh:Mesh3D, mesh_name:str) -> None:
        """Compute the projections of the mesh to the faces of the bounding cuboid and show them in the scene.
        
        Args:
            - mesh : The mesh
            - mesh_name : The name of the mesh"""

        start = time.time()

        if f"xy_back_{mesh_name}" not in self.projections:
            back = self.get_projections(mesh, mesh_name)[0]
        else:
            back = self.projections[f"xy_back_{mesh_name}"]
        
        if f"xz_top_{mesh_name}" not in self.projections:
            top = self.get_projections(mesh, mesh_name)[1]
        else:
            top = self.projections[f"xz_top_{mesh_name}"]

        if f"yz_right_{mesh_name}" not in self.projections:
            right = self.get_projections(mesh, mesh_name)[2]
        else:
            right = self.projections[f"yz_right_{mesh_name}"]

        end = time.time()
        self.projections_t[mesh_name] = end - start

        # Add the Projections to the scene
        self.addShape(back, f"xy_back_{mesh_name}")
        self.addShape(top, f"xz_top_{mesh_name}")
        self.addShape(right, f"yz_right_{mesh_name}")

    # COLLISION DETECTION FUNCTIONS
    def collision_detection_aabbs(self, mesh1:Mesh3D, mesh1_name:str, mesh2:Mesh3D, mesh2_name:str, vis:bool = True) -> bool:
        '''Collision detection using the AABBs.
        
        Args:
            - mesh1: The first mesh
            - mesh1_name: The name of the first mesh
            - mesh2: The second mesh
            - mesh2_name: The name of the second mesh
            - vis: Whether to visualize the intersecting cuboid or not
            
        Returns:
            - inter: True if the meshes intersect, False otherwise'''

        start = time.time()
        aabb1 = self.get_aabb(mesh1, mesh1_name)
        aabb2 = self.get_aabb(mesh2, mesh2_name)
 
        # Find the intersecting cuboid of the two AABBs
        inter_min, inter_max = U.intersect_cuboids(aabb1, aabb2)
        inter = False
        if inter_min is not None and inter_max is not None:
            inter = True
            
            if vis:
                # Visualize the intersecting cuboid

                # A label for the intersecting cuboid
                # label = Label3D(inter_min, f"inter_cuboid_{mesh1_name}_{mesh2_name}", color=Color.BLACK)
                # self.addShape(label, f"label_inter_cuboid_{mesh1_name}_{mesh2_name}")
                # self.misc_geometries[f"label_inter_cuboid_{mesh1_name}_{mesh2_name}"] = label

                # The intersecting cuboid
                inter_cuboid = Cuboid3D(p1=inter_min, p2=inter_max, color=Color.CYAN, filled=False)
                self.addShape(inter_cuboid, f"inter_cuboid_{mesh1_name}_{mesh2_name}")
                self.collision_aabbs[f"inter_cuboid_{mesh1_name}_{mesh2_name}"] = inter_cuboid
        
        end = time.time()
        self.aabb_col_t[f"{mesh1_name}_{mesh2_name}"] = end - start
        
        return inter

    def collision_detection_chs(self, mesh1:Mesh3D, mesh_name1:str, mesh2:Mesh3D, mesh_name2:str) -> bool:
        '''Collision detection using the Convex Hulls.
        
        Args:
            - mesh1: The first mesh
            - mesh_name1: The name of the first mesh
            - mesh2: The second mesh
            - mesh_name2: The name of the second mesh
            
        Returns:
            - True if the meshes intersect, False otherwise'''
        
        start = time.time()
        ch1 = self.get_convex_hull(mesh1, mesh_name1)
        ch2 = self.get_convex_hull(mesh2, mesh_name2)

        col = U.collision(ch1, ch2)
        end = time.time()
        self.ch_col_t[f"{mesh1}_{mesh2}"] = end - start
        return col

    def collision_detection_kdops(self, mesh1:Mesh3D, mesh_name1:str, mesh2:Mesh3D, mesh_name2:str) -> bool:
        '''Collision detection using the 14-DOPs.
        
        Args:
            - mesh1: The first mesh
            - mesh_name1: The name of the first mesh
            - mesh2: The second mesh
            - mesh_name2: The name of the second mesh
            
        Returns:
            - True if the meshes intersect, False otherwise'''

        start = time.time()
        # Get the minimum and maximum extents of the 14-DOPs
        min1, max1, _ = U.get_min_max_directions(mesh1, DIRECTIONS)
        min2, max2, _ = U.get_min_max_directions(mesh2, DIRECTIONS)

        k = len(min1)  # Number of axes (directions)

        col = False
        # Check if there is any separation along any of the k axes
        for i in range(k):
            if max1[i] < min2[i] or max2[i] < min1[i]:
                col = False  # No intersection if separated along this axis
                break
            col = True
        
        end = time.time()
        self.kdop_col_t[f"{mesh_name1}_{mesh_name2}"] = end - start
        
        return col
    
    def collision_detection_meshes(self, mesh1:Mesh3D, mesh_name1:str, mesh2:Mesh3D, mesh_name2:str, vis:bool=True) -> bool:
        '''Collision detection using the meshes themselves and optionally visualizing the points.
        
        Args:
            - mesh1: The first mesh
            - mesh_name1: The name of the first mesh
            - mesh2: The second mesh
            - mesh_name2: The name of the second mesh
            - vis: Whether to visualize the collision points or not 
            
        Returns:
            - True if the meshes intersect, False otherwise'''
        
        start = time.time()
        # Check for collisions using the fastest (and least accurate) methods first
        # Least accurate and fastest
        # projections_collision = self.collision_detection_projections(mesh1, mesh_name1, mesh2, mesh_name2)
        # if not projections_collision:
        #     return False
        
        aabb_collision = self.collision_detection_aabbs(mesh1, mesh_name1, mesh2, mesh_name2, False)
        if not aabb_collision:
            return False
        
        # Occasionally crashes
        kdop_collision = self.collision_detection_kdops(mesh1, mesh_name1, mesh2, mesh_name2)
        if not kdop_collision:
            return False
        
        ch_collision = self.collision_detection_chs(mesh1, mesh_name1, mesh2, mesh_name2)
        if not ch_collision:
            return False
        
        # Most accurate and slowest
        mesh_collision, points = U.collision(mesh1, mesh2, return_points=True)
        if not mesh_collision:
            return False
                
        end = time.time()
        self.mesh_col_t[f"{mesh_name1}_{mesh_name2}"] = end - start
        if vis:
            # Add the collision points to the scene
            for i, point in enumerate(points):
                point = Point3D(point, color=Color.CYAN, size=1)
                self.collision_points[f"col_point_{mesh_name1}_{mesh_name2}_{i}"] = point
                self.addShape(point, f"col_point_{mesh_name1}_{mesh_name2}_{i}")
            
        # Check for collision
        return mesh_collision
    
    def collision_detection_projections(self, mesh1:Mesh3D, mesh_name1:str, mesh2:Mesh3D, mesh_name2:str) -> bool:
        '''Collision detection using the projections.
        
        Args:
            - mesh1: The first mesh
            - mesh_name1: The name of the first mesh
            - mesh2: The second mesh
            - mesh_name2: The name of the second mesh
            
        Returns:
            - True if the meshes MIGHT collide, False otherwise'''

        start = time.time()
        back1, top1, right1 = self.get_projections(mesh1, mesh_name1)
        back2, top2, right2 = self.get_projections(mesh2, mesh_name2)
        # Check for collisions in each face
        col1 = U.collision(back1, back2)
        col2 = U.collision(top1, top2)
        col3 = U.collision(right1, right2)

        # if they collide in all the faces, then they MIGHT collide
        col = col1 and col2 and col3

        end = time.time()
        self.projections_col_t[f"{mesh_name1}_{mesh_name2}"] = end - start

        return col
    
    # SIMULATION FUNCTIONS
    def on_idle(self):
        '''The idle function of the scene.'''

        # Time check to update the scene
        if time.time() - self.last_update < self.dt:
            return
        
        # Simulation with the drones moving and stopping if they collide
        if not self.paused:
            if self.meshes:
                self.simulate()
                return
            self.show_drones(self.num_of_drones, rand_rot=False, label=False)
            return True
        
        # Simulation with the drones moving and changing their speed to avoid collisions
        if not self.paused_no_collisions:
            if self.meshes:
                # If any collision points exist, remove them
                if self.collision_points:
                    for collision_point_name in self.collision_points.keys():
                        self.removeShape(collision_point_name)
                    self.collision_points = {}
                self.simulate_no_collisions()
                return
            self.show_drones(self.num_of_drones, rand_rot=False, label=False)
            return True
        
        # Landing Protocol
        if not self.pause_landing_simulation:
            if self.meshes:
                # If any collision points exist, remove them
                if self.collision_points:
                    for collision_point_name in self.collision_points.keys():
                        self.removeShape(collision_point_name)
                    self.collision_points = {}
                self.landing_protocol()
                return
            self.show_drones(self.num_of_drones, rand_rot=False, label=False)
        
        return False
    
    def move_drone(self, mesh:Mesh3D, mesh_name:str, speed:np.ndarray, label_f:bool=False) -> None:
        '''Move the drone and its projections with a certain speed.
        
        Args:
            - mesh : The mesh
            - mesh_name : The name of the mesh
            - speed : The speed of the drone
            - label_f : Whether to move the label or not
        '''
        self.last_update = time.time() 
        translation_vector = speed
        
        ## Move the drone
        mesh.vertices += translation_vector
        self.updateShape(mesh_name, quick=True)

        if label_f:
            # Move the label
            label = self.misc_geometries[f"label_{mesh_name}"]
            label.x += translation_vector[0]
            label.y += translation_vector[1]
            label.z += translation_vector[2]
            self.updateShape(f"label_{mesh_name}", quick=True)


        ## Move the projections
        # Get the projections
        back, top, right = self.get_projections(mesh, mesh_name)
        
        # Move the projections in the 2D planes
        # Back face
        back.vertices += [translation_vector[0], translation_vector[1], 0]
        self.updateShape(f"xy_back_{mesh_name}", quick=True)

        # Top face
        top.vertices += [translation_vector[0], 0, translation_vector[2]]
        self.updateShape(f"xz_top_{mesh_name}", quick=True)

        # Right face
        right.vertices += [0, translation_vector[1], translation_vector[2]]
        self.updateShape(f"yz_right_{mesh_name}", quick=True)

    def move_drone_to_point(self, mesh:Mesh3D, mesh_name:str, point:np.ndarray, label_f:bool=False) -> None:
        '''Move the drone to a point.
        
        Args:
            - mesh : The mesh
            - mesh_name : The name of the mesh
            - point : The point
        '''
        self.last_update = time.time() 
        # Move the drone
        mesh = U.shift_center_of_mass(mesh, point)
        self.updateShape(mesh_name)
        if label_f:
            # Move the label
            label = self.misc_geometries[f"label_{mesh_name}"]
            label.x = point[0]
            label.y = point[1]
            label.z = point[2]
            self.updateShape(f"label_{mesh_name}")

    def simulate(self):
        '''Simulate the scene, moving the drones and stopping them if they collide.'''


        # Remove any aabbs, chs, kdops, collision points and labels from the scene
        self.clear_attributes()

        for mesh_name, mesh in self.meshes.items():

            # If all the drones have stopped, clear the scene
            if self.moving_meshes == {}:
                self.reset_scene()
                return
            
            # If the drone is out of bounds, remove it
            if not self.bounding_cuboid.check_mesh_in_cuboid(mesh):

                # self.print(f"-Drone {mesh_name} out of bounds")
                if mesh_name in self.moving_meshes:
                    self.moving_meshes.pop(mesh_name)
                    continue

                self.removeShape(mesh_name)
                continue

            if mesh_name in self.moving_meshes:
                self.move_drone(mesh, mesh_name, self.speeds[mesh_name]) 

    def simulate_no_collisions(self):
        '''Simulate the scene , moving the dornes and changing their speed if they collide to avoid collisions.'''

        # Remove any aabbs, chs, kdops, collision points and labels from the scene
        self.clear_attributes()

        for mesh_name, mesh in self.meshes.items():

            # If all the drones have stopped, clear the scene
            if self.moving_meshes == {}:
                self.reset_scene()
                return
            
            # If the drone is out of bounds, remove it
            if not self.bounding_cuboid.check_mesh_in_cuboid(mesh):
                # self.print(f"-Drone {mesh_name} out of bounds")
                if mesh_name in self.moving_meshes:
                    self.moving_meshes.pop(mesh_name)
                    continue

                self.removeShape(mesh_name)
                continue
            
            self.move_drone(mesh, mesh_name, self.speeds[mesh_name]) 
                  
            for j, (mesh_name2, mesh2) in enumerate(self.meshes.items()):

                # Skip the same mesh
                if mesh_name > mesh_name2:

                    # # Get a copy of the drone in the next frame
                    # mesh_copy = mesh.get_copy()
                    # mesh_copy.vertices += self.speeds[mesh_name]

                    # # Check if the drone will collide with the other drone in the next frame
                    # if self.collision_detection_meshes(mesh_copy, mesh_name, mesh2, mesh_name2):
                    #     self.print(f"-Mesh3D collision between {mesh_name} and {mesh_name2}")

                    # Check for collisions and adjust the speed to avoid them (dont check the next frame)
                    if self.collision_detection_meshes(mesh, mesh_name, mesh2, mesh_name2, vis=False):
                        self.print(f"-Mesh3D collision between {mesh_name} and {mesh_name2}")

                        ## DOESNT WORK IT CRASHES
                        # # Calculate the surface normal of the collision point
                        # collision_point = self.collision_points[f"col_point_{mesh_name}_{mesh_name2}_0"]
                        # collision_point = np.array([collision_point.x, collision_point.y, collision_point.z])
                        # surface_normal = U.get_surface_normal(mesh, collision_point)

                        # # Change the speed of the drone to avoid collisions
                        # speed1 = self.speeds[mesh_name]

                        # # Change the speed of the drones to avoid collisions
                        # new_speed1 = speed1 - 2 * np.dot(speed1, surface_normal) * surface_normal
                        # print(new_speed1)
                        # print(mesh_name)
                        # self.speeds[mesh_name] = new_speed1
                        # continue

                        # Get the current speeds of the drones
                        speed1 = self.speeds[mesh_name]
                        speed2 = self.speeds[mesh_name2]

                        # Reverse the speeds of the drones
                        new_speed1 = -1 * speed1
                        new_speed2 = -1 * speed2

                        # Update the speeds of the drones
                        self.speeds[mesh_name] = new_speed1
                        self.speeds[mesh_name2] = new_speed2
                                                 
    def landing_protocol(self):
        '''Simulate the landing protocol.'''
        self.last_update = time.time()

        # Remove any aabbs, chs, kdops, collision points and labels from the scene
        self.clear_attributes()
        
        # Spawn drones at random time intervals until there are N^2 drones (one drone per landing pad)
        if not len(self.meshes) == self.N**2: 
            if time.time() - self.last_spawn_time > self.spawn_interval:
                self.show_drones(1, rand_rot=False, singular=True)
                # self.print(f"Drone has spawned")
                self.last_spawn_time = time.time()

        for i, (mesh_name, mesh) in enumerate(self.meshes.items()):


            landing_pad = list(self.landing_pads.values())[i]
            landing_point = landing_pad.get_center() + np.array([0, 0.3, 0]) # adjust the landing point so that the drone is above the landing pad, not inside it


            # Calculate the distance between the drone and the landing pad
            distance = np.linalg.norm(mesh.get_center() - landing_point)

            # Move the drone to the direction of the landing pad
            if distance > 0.1:

                # Calculate the direction to the landing pad
                direction = (landing_point - mesh.get_center(lst=True))

                # Calculate the speed
                speed_modifier = np.random.uniform(0.1, 0.3) # Random speed
                speed = direction / np.linalg.norm(direction) * speed_modifier
                self.move_drone(mesh, mesh_name, speed)

                # Check for collisions and adjust the speed to avoid them
                for j, (mesh_name2, mesh2) in enumerate(self.meshes.items()):
                    if mesh_name > mesh_name2:
                        if self.collision_detection_meshes(mesh, mesh_name, mesh2, mesh_name2, vis=False):
                            self.print(f"-Mesh3D collision between {mesh_name} and {mesh_name2}")

                            # Change the vertical speed of the drone to avoid collisions
                            speed[2] += 0.1
                return
            
            # If the drone is close to the landing pad, move it to the landing pad
            if distance <= 0.1:
                self.move_drone_to_point(mesh, mesh_name, landing_point)
                
                # Check if the drone has landed
                if mesh_name not in self.landed_meshes:
                    self.print(f"{mesh_name} has landed")
                self.landed_meshes[mesh_name] = mesh

                # If all the drones have landed, pause the landing simulation
                if len(self.landed_meshes) == len(self.meshes):
                    self.pause_landing_simulation = True 
                continue

    # STATS FUNCTION
    def stats(self):
        """Print the statistics of the scene."""

        # Print the statistics of the scene
        self.print("----------------------------------------")
        self.print("Statistics:")
        self.print("----------------------------------------")
        self.print(f"-Number of drones: {len(self.meshes)}")
        self.print(f"---Time to create the Convex Hulls: {np.sum(list(self.ch_t.values())):.4f} seconds")
        self.print(f"---Time to create the AABBs: {np.sum(list(self.aabb_t.values())):.4f} seconds")
        self.print(f"---Time to create the 14-DOPs: {np.sum(list(self.kdop_t.values())):.4f} seconds")
        self.print(f"---Time to create the Projections: {np.sum(list(self.projections_t.values())):.4f} seconds")
        self.print(f"---Time to check for AABB collisions: {np.sum(list(self.aabb_col_t.values())):.4f} seconds")
        self.print(f"---Time to check for Convex Hull collisions: {np.sum(list(self.ch_col_t.values())):.4f} seconds")
        self.print(f"---Time to check for 14-DOP collisions: {np.sum(list(self.kdop_col_t.values())):.4f} seconds")
        self.print(f"---Time to check for Mesh3D collisions: {np.sum(list(self.mesh_col_t.values())):.4f} seconds")
        self.print(f"---Time to check for Projections collisions: {np.sum(list(self.projections_col_t.values())):.4f} seconds")
        self.print("----------------------------------------")

if __name__ == "__main__":
    scene = UavSim()
    scene.mainLoop()








