# create_mesh.py
# This script generates a 2D mesh for a photodiode using Gmsh.
# The output file "output/photodiode_mesh.msh" is designed to be used
# as the input for the provided devsim_simulation.py script.
#
# CORRECTION V4:
# Added a specific mesh size parameter for contacts ("MESH_CONTACT") to
# provide more direct control over the mesh density at the cathode,
# ensuring adequate resolution without making the entire bulk region too dense.

import gmsh
import os
import sys

# --- Simulation Parameters ---
# All dimensions are in micrometers (um).
WIDTH = 10.0  # Total width of the device
DEPTH = 5.0   # Total depth of the device

P_WIDTH = 4.0 # Width of the p-region
P_DEPTH = 1.0 # Depth of the p-region

# --- Gap Parameter ---
# Defines the size of the insulating gap between the contact and junction.
GAP_SIZE = 0.1 # um

# --- NEW: Smart Mesh Density ---
# Use a fine mesh near the junction and a coarse mesh elsewhere.
MESH_FINE = 0.05    # Finer mesh size for the active junction area
MESH_CONTACT = 0.15  # A specific, medium mesh size for the contacts
MESH_COARSE = 0.3   # Coarser mesh size for non-critical corners

# --- Output Directory and File ---
OUTPUT_DIR = "output"
MESH_FILE = os.path.join(OUTPUT_DIR, "photodiode_mesh.msh")

# ==============================================================================
#                      GMSH SCRIPT
# ==============================================================================

# Initialize Gmsh
gmsh.initialize(sys.argv)
gmsh.model.add("photodiode")

print("Starting smart mesh generation for 2D photodiode...")

# --- 1. Define Geometry Points ---
# We now assign mesh sizes on a per-point basis. Gmsh will automatically
# create a graded mesh that transitions between the different sizes.

# Coarse points for the non-contact corners
p1 = gmsh.model.geo.addPoint(0,         0,         0, MESH_COARSE)
p4 = gmsh.model.geo.addPoint(WIDTH,     0,         0, MESH_COARSE)

# Points for the cathode contact
p7 = gmsh.model.geo.addPoint(0,         -DEPTH,    0, MESH_CONTACT)
p8 = gmsh.model.geo.addPoint(WIDTH,     -DEPTH,    0, MESH_CONTACT)

# Fine points defining the p-n junction and active area
p_x_start = (WIDTH - P_WIDTH) / 2.0
p_x_end = p_x_start + P_WIDTH
p2 = gmsh.model.geo.addPoint(p_x_start, 0,         0, MESH_FINE)
p3 = gmsh.model.geo.addPoint(p_x_end,   0,         0, MESH_FINE)
p5 = gmsh.model.geo.addPoint(p_x_start, -P_DEPTH,  0, MESH_FINE)
p6 = gmsh.model.geo.addPoint(p_x_end,   -P_DEPTH,  0, MESH_FINE)

# Fine points for the anode gaps
p_anode_start = gmsh.model.geo.addPoint(p_x_start + GAP_SIZE, 0, 0, MESH_FINE)
p_anode_end   = gmsh.model.geo.addPoint(p_x_end - GAP_SIZE,   0, 0, MESH_FINE)

print(f"Defined 10 geometric points with fine, contact, and coarse mesh sizes.")

# --- 2. Define Lines from Points ---
# The top surface is now broken into 5 segments
l_top_left = gmsh.model.geo.addLine(p1, p2)
l_gap1 = gmsh.model.geo.addLine(p2, p_anode_start) # Left gap
l_anode = gmsh.model.geo.addLine(p_anode_start, p_anode_end) # Anode contact
l_gap2 = gmsh.model.geo.addLine(p_anode_end, p3) # Right gap
l_top_right = gmsh.model.geo.addLine(p3, p4)

# Other device boundaries
l_right_wall = gmsh.model.geo.addLine(p4, p8)
l_cathode = gmsh.model.geo.addLine(p8, p7)
l_left_wall = gmsh.model.geo.addLine(p7, p1)

# Inner p-region boundary lines
l_p_right = gmsh.model.geo.addLine(p3, p6)
l_p_bottom = gmsh.model.geo.addLine(p6, p5)
l_p_left = gmsh.model.geo.addLine(p5, p2)

print("Defined boundary lines.")

# --- 3. Define Surfaces from Lines ---
# Create curve loops (closed paths of lines)
cl_outer = gmsh.model.geo.addCurveLoop([l_top_left, l_gap1, l_anode, l_gap2, l_top_right, l_right_wall, l_cathode, l_left_wall])
cl_p_region = gmsh.model.geo.addCurveLoop([l_gap1, l_anode, l_gap2, l_p_right, l_p_bottom, l_p_left])

# Create plane surfaces from the curve loops.
s_n_region = gmsh.model.geo.addPlaneSurface([cl_outer, cl_p_region])
s_p_region = gmsh.model.geo.addPlaneSurface([cl_p_region])

print("Defined p-region and n-region surfaces.")

# Synchronize the CAD model before defining physical groups
gmsh.model.geo.synchronize()

# --- 4. Define Physical Groups ---
# Physical Surfaces (2D entities)
gmsh.model.addPhysicalGroup(2, [s_p_region], name="p_region")
gmsh.model.addPhysicalGroup(2, [s_n_region], name="n_region")

# Physical Lines (1D entities)
gmsh.model.addPhysicalGroup(1, [l_anode], name="anode")
gmsh.model.addPhysicalGroup(1, [l_cathode], name="cathode")

# The pn_junction remains the sides and bottom of the p-region.
pn_junction_lines = [l_p_right, l_p_bottom, l_p_left]
gmsh.model.addPhysicalGroup(1, pn_junction_lines, name="pn_junction")

print("Assigned final physical groups.")

# --- 5. Generate and Save Mesh ---
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)

gmsh.option.setNumber("Mesh.Algorithm", 6) # 6 = Frontal-Delaunay

gmsh.option.setNumber("Mesh.Optimize", 1)

gmsh.model.mesh.generate(2)
print("2D smart mesh generated successfully.")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

gmsh.write(MESH_FILE)
print(f"Mesh file saved to: {MESH_FILE}")

gmsh.finalize()
