# Drone Simulation App

This project is a simulation environment for drones, providing a variety of features including random positioning, collision detection, and visualization toggles.

## Requirements

Make sure you have the following prerequisites:

- Python version `3.10.13` or later
- Conda package manager

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Papiqulos/Project_3D_UAV_2024
```

### 2. Create and Activate a VIrtual Environment (Conda Recommended)

Use the following commands to create and activate a new conda environment:

```bash
conda create --name drone-sim python=3.10.13
conda activate drone-sim
```

### 3. Install the Required Dependencies

Install the necessary Python packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Running the Application

Once the setup is complete, you can run the drone simulation app. Below are the available commands that work within the app:

### Key Commands

- `R`: Clear scene
- `B`: Toggle bounds
- `S`: Show drones in random positions
- `P`: Show drones in random positions and orientations
- `C`: Toggle convex hulls
- `A`: Toggle AABBs (Axis-Aligned Bounding Boxes)
- `K`: Toggle k-DOPs
- `I`: Toggle projections (xy, xz, yz)
- `N`: Check collisions (AABBs)
- `L`: Check collisions (Convex Hulls)
- `M`: Check collisions (14-DOPs)
- `V`: Check collisions and show collision points (Mesh3Ds)
- `O`: Check collisions (Projections)
- `T`: Simulate
- `F`: Simulate without collisions
- `Q`: Show statistics
- `SPACE`: Trigger the landing protocol
