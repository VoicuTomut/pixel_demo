# create_mesh.py
# Mesh generation script for front-side illuminated (FSI) pixel photodiode

import gmsh
import os
import sys

# --- Simulation Parameters ---
# All dimensions are in micrometers (um).
WIDTH = 5.0   # Pixel width (um)
DEPTH = 5.0   # Pixel depth (um)

P_WIDTH = 4.0   # Width of the p+ region (contact implant)
P_DEPTH = 0.05  # Depth of the p+ region = 50 nm (shallow implant)

# --- Gap Parameter ---
# Defines the size of the insulating gap between the contact and junction.
GAP_SIZE = 0.05  # 50 nm gap, prevents full-metal coverage

# --- Mesh Density ---
MESH_FINE = 0.01     # 10 nm near junction
MESH_CONTACT = 0.05  # 50 nm near contacts
MESH_COARSE = 0.2    # 200 nm in bulk

# --- Output Directory and File ---
OUTPUT_DIR = "output"
MESH_FILE = os.path.join(OUTPUT_DIR, "photodiode_mesh.msh")

# ==============================================================================
#                      GMSH SCRIPT
# ==============================================================================

gmsh.initialize(sys.argv)
gmsh.model.add("photodiode")

print("Starting mesh generation for FSI photodiode...")
print(f"Mesh params: Fine={MESH_FINE}µm, Contact={MESH_CONTACT}µm, Coarse={MESH_COARSE}µm")

# --- 1. Define Geometry Points ---
# Outer corners
p1 = gmsh.model.geo.addPoint(0, 0, 0, MESH_COARSE)
p4 = gmsh.model.geo.addPoint(WIDTH, 0, 0, MESH_COARSE)
p7 = gmsh.model.geo.addPoint(0, -DEPTH, 0, MESH_CONTACT)
p8 = gmsh.model.geo.addPoint(WIDTH, -DEPTH, 0, MESH_CONTACT)

# P+ region (shallow junction)
p_x_start = (WIDTH - P_WIDTH) / 2.0
p_x_end   = p_x_start + P_WIDTH
p2 = gmsh.model.geo.addPoint(p_x_start, 0, 0, MESH_FINE)
p3 = gmsh.model.geo.addPoint(p_x_end,   0, 0, MESH_FINE)
p5 = gmsh.model.geo.addPoint(p_x_start, -P_DEPTH, 0, MESH_FINE)
p6 = gmsh.model.geo.addPoint(p_x_end,   -P_DEPTH, 0, MESH_FINE)

# Anode contact (with small insulating gaps)
p_anode_start = gmsh.model.geo.addPoint(p_x_start + GAP_SIZE, 0, 0, MESH_FINE)
p_anode_end   = gmsh.model.geo.addPoint(p_x_end   - GAP_SIZE, 0, 0, MESH_FINE)

print("Defined geometry points with shallow p+ implant.")

# --- 2. Define Lines ---
l_top_left   = gmsh.model.geo.addLine(p1, p2)
l_gap1       = gmsh.model.geo.addLine(p2, p_anode_start)
l_anode      = gmsh.model.geo.addLine(p_anode_start, p_anode_end)
l_gap2       = gmsh.model.geo.addLine(p_anode_end, p3)
l_top_right  = gmsh.model.geo.addLine(p3, p4)

l_right_wall = gmsh.model.geo.addLine(p4, p8)
l_cathode    = gmsh.model.geo.addLine(p8, p7)
l_left_wall  = gmsh.model.geo.addLine(p7, p1)

# P+ region boundaries
l_p_right  = gmsh.model.geo.addLine(p3, p6)
l_p_bottom = gmsh.model.geo.addLine(p6, p5)
l_p_left   = gmsh.model.geo.addLine(p5, p2)

# --- 3. Define Surfaces ---
cl_outer = gmsh.model.geo.addCurveLoop([l_top_left, l_gap1, l_anode, l_gap2, l_top_right,
                                        l_right_wall, l_cathode, l_left_wall])
cl_p_region = gmsh.model.geo.addCurveLoop([l_gap1, l_anode, l_gap2, l_p_right,
                                           l_p_bottom, l_p_left])

s_n_region = gmsh.model.geo.addPlaneSurface([cl_outer, cl_p_region])
s_p_region = gmsh.model.geo.addPlaneSurface([cl_p_region])

gmsh.model.geo.synchronize()

# --- 4. Physical Groups ---
gmsh.model.addPhysicalGroup(2, [s_p_region], name="p_region")
gmsh.model.addPhysicalGroup(2, [s_n_region], name="n_region")
gmsh.model.addPhysicalGroup(1, [l_anode],    name="anode")
gmsh.model.addPhysicalGroup(1, [l_cathode],  name="cathode")
gmsh.model.addPhysicalGroup(1, [l_p_right, l_p_bottom, l_p_left], name="pn_junction")

print("Assigned physical groups (p, n, anode, cathode, junction).")

# --- 5. Mesh Generation ---
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
gmsh.option.setNumber("Mesh.Algorithm", 6) # Frontal-Delaunay
gmsh.option.setNumber("Mesh.Optimize", 1)

gmsh.model.mesh.generate(2)

# Report
_, node_tags, _ = gmsh.model.mesh.getNodes()
element_types, element_tags, _ = gmsh.model.mesh.getElements()
num_elements = sum(len(tags) for tags in element_tags)
print(f"Mesh stats: {len(node_tags)} nodes, {num_elements} elements")

# Save mesh
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
gmsh.write(MESH_FILE)

gmsh.finalize()
print(f"✅ Mesh saved to {MESH_FILE}")
