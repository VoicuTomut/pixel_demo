# create_mesh.py
# This script generates a 2D mesh for a photodiode using Gmsh.
# The output file "output/photodiode_mesh.msh" is designed to be used
# as the input for the provided devsim_simulation.py script.
#
# CORRECTION:
# The physical groups for contacts and the junction have been redefined
# to ensure contacts are correctly placed on the boundaries of their
# respective regions, which resolves the DEVSIM error.

import gmsh
import os
import sys

# --- Simulation Parameters ---
# All dimensions are in micrometers (um).
WIDTH = 10.0  # Total width of the device
DEPTH = 5.0   # Total depth of the device

P_WIDTH = 4.0 # Width of the p-region
P_DEPTH = 1.0 # Depth of the p-region

# --- Mesh Density ---
# A smaller value creates a finer (more detailed) mesh.
MESH_SIZE = 0.1

# --- Output Directory and File ---
OUTPUT_DIR = "output"
MESH_FILE = os.path.join(OUTPUT_DIR, "photodiode_mesh.msh")

# ==============================================================================
#                      GMSH SCRIPT
# ==============================================================================

# Initialize Gmsh
gmsh.initialize(sys.argv)
gmsh.model.add("photodiode")

print("Starting mesh generation for 2D photodiode...")

# --- 1. Define Geometry Points ---
# Geometry is centered for simplicity.
p_x_start = (WIDTH - P_WIDTH) / 2.0
p_x_end = p_x_start + P_WIDTH

# Add points for the geometry
# Format: gmsh.model.geo.addPoint(x, y, z, meshSize, tag)
# Note: z=0 for all points in a 2D simulation.
p1 = gmsh.model.geo.addPoint(0,         0,         0, MESH_SIZE)
p2 = gmsh.model.geo.addPoint(p_x_start, 0,         0, MESH_SIZE)
p3 = gmsh.model.geo.addPoint(p_x_end,   0,         0, MESH_SIZE)
p4 = gmsh.model.geo.addPoint(WIDTH,     0,         0, MESH_SIZE)
p5 = gmsh.model.geo.addPoint(p_x_start, -P_DEPTH,  0, MESH_SIZE)
p6 = gmsh.model.geo.addPoint(p_x_end,   -P_DEPTH,  0, MESH_SIZE)
p7 = gmsh.model.geo.addPoint(0,         -DEPTH,    0, MESH_SIZE)
p8 = gmsh.model.geo.addPoint(WIDTH,     -DEPTH,    0, MESH_SIZE)

print(f"Defined 8 geometric points.")

# --- 2. Define Lines from Points ---
# These lines form the boundaries of the regions.
# Format: gmsh.model.geo.addLine(start_point_tag, end_point_tag, tag)
l_top_left = gmsh.model.geo.addLine(p1, p2)
l_anode = gmsh.model.geo.addLine(p2, p3) # This is the top of the p-region
l_top_right = gmsh.model.geo.addLine(p3, p4)
l_right_wall = gmsh.model.geo.addLine(p4, p8)
l_cathode = gmsh.model.geo.addLine(p8, p7) # This is the bottom of the n-region
l_left_wall = gmsh.model.geo.addLine(p7, p1)

# Inner p-region boundary lines
l_p_right = gmsh.model.geo.addLine(p3, p6)
l_p_bottom = gmsh.model.geo.addLine(p6, p5)
l_p_left = gmsh.model.geo.addLine(p5, p2)

print("Defined 9 boundary lines.")

# --- 3. Define Surfaces from Lines ---
# Create curve loops (closed paths of lines)
cl_outer = gmsh.model.geo.addCurveLoop([l_top_left, l_anode, l_top_right, l_right_wall, l_cathode, l_left_wall])
cl_p_region = gmsh.model.geo.addCurveLoop([l_anode, l_p_right, l_p_bottom, l_p_left])

# Create plane surfaces from the curve loops.
# The n-region is the outer surface with the p-region cut out as a hole.
s_n_region = gmsh.model.geo.addPlaneSurface([cl_outer, cl_p_region])
s_p_region = gmsh.model.geo.addPlaneSurface([cl_p_region])

print("Defined p-region and n-region surfaces.")

# Synchronize the CAD model before defining physical groups
gmsh.model.geo.synchronize()

# --- 4. Define Physical Groups (Corrected) ---
# This step assigns names that DEVSIM will use. The names must
# exactly match the 'gmsh_name' in your simulation script.

# Physical Surfaces (2D entities)
gmsh.model.addPhysicalGroup(2, [s_p_region], name="p_region")
gmsh.model.addPhysicalGroup(2, [s_n_region], name="n_region")

# Physical Lines (1D entities)
# The anode is now the top of the p-region.
gmsh.model.addPhysicalGroup(1, [l_anode], name="anode")
# The cathode is now the bottom of the n-region.
gmsh.model.addPhysicalGroup(1, [l_cathode], name="cathode")

# The pn_junction is where the p-region and n-region touch.
# This is the sides and bottom of the p-region rectangle.
pn_junction_lines = [l_p_right, l_p_bottom, l_p_left]
gmsh.model.addPhysicalGroup(1, pn_junction_lines, name="pn_junction")

print("Assigned corrected physical groups for regions, contacts, and the junction.")

# --- 5. Generate and Save Mesh ---
# Set Gmsh options for mesh file format (Version 2 is widely compatible)
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)

# Generate the 2D mesh
gmsh.model.mesh.generate(2)
print("2D mesh generated successfully.")

# Create the output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Write the mesh file
gmsh.write(MESH_FILE)
print(f"Mesh file saved to: {MESH_FILE}")

# Finalize Gmsh
gmsh.finalize()
