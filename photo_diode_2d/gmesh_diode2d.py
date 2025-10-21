#
# Gmsh-based Photodiode Mesh Generator for DEVSIM
#
# Creates a vertical p-n junction photodiode with corrected physical group names
# and comprehensive mesh refinement for accurate DEVSIM TCAD simulation.
#
"""
      Light ↓ (incident on y=0 surface)
      <-- width -->
      +-----------+
      |  cathode  |
+-----+-----------+-----+  y = 0
|     n+ region         |  (n_plus_region)
|-----------------------|  y = n_plus_thickness
|                       |
|       p region        |  (p_region)
|                       |
|-----------------------|  y = n_plus_thickness + p_thickness
|     p+ substrate      |  (p_plus_region)
+-----------------------+  y = total_depth
|         anode         |
+-----------------------+
"""

import gmsh
import sys
import numpy as np


class PhotodiodeMesh:
    """
    Parametric photodiode mesh generator optimized for DEVSIM.


    All dimensions are input in micrometers (μm) and are converted to
    centimeters (cm) internally, as required by DEVSIM.
    """

    def __init__(self,
                 # Device dimensions [μm]
                 width=100.0,
                 n_plus_thickness=0.5,
                 p_thickness=50.0,
                 p_plus_thickness=10.0,

                 # Contact dimensions [μm]
                 top_contact_x_start=45.0,
                 top_contact_x_end=55.0,

                 # Mesh refinement [μm] - ALL ARE NOW USED
                 # mesh_size_junction=0.002,  # 2 nm at junction
                 # mesh_size_interface=0.05,  # 50 nm at p/p+ interface
                 # mesh_size_surface=0.02,  # 20 nm at non-contact surfaces
                 # mesh_size_contact=0.02,  # 20 nm at contacts
                 # mesh_size_bulk=2.0,  # 2 μm in bulk regions

                 mesh_size_junction=0.05,  # Was 0.002 (2 nm), now 50 nm
                 mesh_size_interface=0.05,  # Was 0.05, now 200 nm
                 mesh_size_surface=0.02,  # Was 0.02, now 500 nm
                 mesh_size_contact=0.02,  # Was 0.02, now 500 nm
                 mesh_size_bulk=1.0,  # Was 2.0, now 5 μm

                 # Output options
                 output_file="gmsh_diode2d.msh",
                 show_gui=True):

        # --- Validate Inputs ---
        if top_contact_x_start < 0 or top_contact_x_end > width:
            raise ValueError(f"Contact must be within device width (0 to {width} μm)")
        if top_contact_x_start >= top_contact_x_end:
            raise ValueError("Contact start must be less than contact end")


        # --- Store parameters and convert units from μm to cm (DEVSIM default) ---
        self.width = width * 1e-4
        self.n_plus_thickness = n_plus_thickness * 1e-4
        self.p_thickness = p_thickness * 1e-4
        self.p_plus_thickness = p_plus_thickness * 1e-4

        self.total_depth = self.n_plus_thickness + self.p_thickness + self.p_plus_thickness
        self.junction_y = self.n_plus_thickness
        self.p_pplus_interface_y = self.n_plus_thickness + self.p_thickness

        self.top_contact_x_start = top_contact_x_start * 1e-4
        self.top_contact_x_end = top_contact_x_end * 1e-4

        # Mesh sizes converted to cm
        self.mesh_size_junction = mesh_size_junction * 1e-4
        self.mesh_size_interface = mesh_size_interface * 1e-4
        self.mesh_size_surface = mesh_size_surface * 1e-4
        self.mesh_size_contact = mesh_size_contact * 1e-4
        self.mesh_size_bulk = mesh_size_bulk * 1e-4

        self.output_file = output_file
        self.show_gui = show_gui

        # Dictionaries to store Gmsh entity tags
        self.curves = {}

    def create_mesh(self):
        """Main method to orchestrate the mesh generation process."""
        gmsh.initialize()
        # Force legacy MSH 2.2 ASCII so DEVSIM can import it reliably
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.Binary", 0)

        gmsh.model.add("photodiode_corrected")

        print("=" * 70)
        print("CREATING CORRECTED PHOTODIODE MESH FOR DEVSIM")
        print("=" * 70)

        self._create_geometry()
        self._set_mesh_sizes()

        print("\nGenerating 2D mesh...")
        gmsh.model.mesh.generate(2)

        print("Optimizing mesh quality...")
        gmsh.model.mesh.optimize("Netgen")

        self._print_statistics()
        self._print_physical_groups()

        print(f"\nSaving mesh to: {self.output_file}")
        gmsh.write(self.output_file)

        if self.show_gui:
            print("\nOpening Gmsh GUI for visualization...")
            gmsh.fltk.run()

        gmsh.finalize()
        print("\n" + "=" * 70)
        print("MESH GENERATION COMPLETE!")
        print(f"File '{self.output_file}' is ready for DEVSIM import.")
        print("=" * 70)

    def _create_geometry(self):
        """Create the photodiode geometry with all regions and boundaries."""
        geo = gmsh.model.geo
        print("\nCreating geometry...")

        # --- Define all corner and interface points (CCW order from top-left) ---
        p1 = geo.addPoint(0, 0, 0)
        p_tc_left = geo.addPoint(self.top_contact_x_start, 0, 0)
        p_tc_right = geo.addPoint(self.top_contact_x_end, 0, 0)
        p2 = geo.addPoint(self.width, 0, 0)
        p3 = geo.addPoint(self.width, self.junction_y, 0)
        p4 = geo.addPoint(self.width, self.p_pplus_interface_y, 0)
        p5 = geo.addPoint(self.width, self.total_depth, 0)
        p6 = geo.addPoint(0, self.total_depth, 0)
        p7 = geo.addPoint(0, self.p_pplus_interface_y, 0)
        p8 = geo.addPoint(0, self.junction_y, 0)

        # --- Create all boundary and interface lines ---
        l_top_left = geo.addLine(p1, p_tc_left)
        l_top_contact = geo.addLine(p_tc_left, p_tc_right)
        l_top_right = geo.addLine(p_tc_right, p2)
        l_right_n = geo.addLine(p2, p3)
        l_right_p = geo.addLine(p3, p4)
        l_right_pplus = geo.addLine(p4, p5)
        l_bottom = geo.addLine(p5, p6)
        l_left_pplus = geo.addLine(p6, p7)
        l_left_p = geo.addLine(p7, p8)
        l_left_n = geo.addLine(p8, p1)

        l_junction = geo.addLine(p8, p3)
        l_interface_p_pplus = geo.addLine(p7, p4)

        self.curves = {
            'cathode': l_top_contact,
            'anode': l_bottom,
            'junction': l_junction,
            'interface_p_pplus': l_interface_p_pplus,
            'top_surface': [l_top_left, l_top_right]
        }

        # --- Create curve loops and surfaces (all CCW) ---
        loop_n_plus = geo.addCurveLoop([l_top_left, l_top_contact, l_top_right, l_right_n, -l_junction, l_left_n])
        surf_n_plus = geo.addPlaneSurface([loop_n_plus])

        loop_p = geo.addCurveLoop([l_junction, l_right_p, -l_interface_p_pplus, l_left_p])
        surf_p = geo.addPlaneSurface([loop_p])

        loop_p_plus = geo.addCurveLoop([l_interface_p_pplus, l_right_pplus, l_bottom, l_left_pplus])
        surf_p_plus = geo.addPlaneSurface([loop_p_plus])

        geo.synchronize()

        # --- Add Physical Groups for DEVSIM ---
        print("\n  Adding physical groups for DEVSIM...")
        gmsh.model.addPhysicalGroup(2, [surf_n_plus], name="n_plus_region")
        gmsh.model.addPhysicalGroup(2, [surf_p], name="p_region")
        gmsh.model.addPhysicalGroup(2, [surf_p_plus], name="p_plus_region")

        # ** CONTACTS **
        # Cathode: Negative terminal, attached to n-type material.
        # Anode:   Positive terminal, attached to p-type material.
        gmsh.model.addPhysicalGroup(1, [l_top_contact], name="cathode")
        gmsh.model.addPhysicalGroup(1, [l_bottom], name="anode")

        gmsh.model.addPhysicalGroup(1, [l_junction], name="pn_interface")
        gmsh.model.addPhysicalGroup(1, [l_interface_p_pplus], name="p_pplus_interface")

        gmsh.model.addPhysicalGroup(1, [l_top_left, l_top_right], name="top_surface")
        gmsh.model.addPhysicalGroup(1, [l_left_n, l_left_p, l_left_pplus], name="left_side")
        gmsh.model.addPhysicalGroup(1, [l_right_n, l_right_p, l_right_pplus], name="right_side")
        print("  ✓ Geometry and physical groups complete!")

    def _set_mesh_sizes(self):
        """Set mesh refinement using multiple Gmsh Fields for precise control."""
        print("\nSetting comprehensive mesh refinement...")

        fields = []

        # Field for p-n junction (most critical)
        field_junction_dist = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(field_junction_dist, "CurvesList", [self.curves['junction']])
        field_junction_thresh = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(field_junction_thresh, "InField", field_junction_dist)
        gmsh.model.mesh.field.setNumber(field_junction_thresh, "SizeMin", self.mesh_size_junction)
        gmsh.model.mesh.field.setNumber(field_junction_thresh, "SizeMax", self.mesh_size_bulk)
        gmsh.model.mesh.field.setNumber(field_junction_thresh, "DistMin", 0.1e-4)
        gmsh.model.mesh.field.setNumber(field_junction_thresh, "DistMax", 5e-4)
        fields.append(field_junction_thresh)
        print(f"  - Junction refinement: {self.mesh_size_junction * 1e7:.1f} nm")

        # Field for contacts
        field_contact_dist = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(field_contact_dist, "CurvesList",
                                         [self.curves['anode'], self.curves['cathode']])
        field_contact_thresh = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(field_contact_thresh, "InField", field_contact_dist)
        gmsh.model.mesh.field.setNumber(field_contact_thresh, "SizeMin", self.mesh_size_contact)
        gmsh.model.mesh.field.setNumber(field_contact_thresh, "SizeMax", self.mesh_size_bulk)
        gmsh.model.mesh.field.setNumber(field_contact_thresh, "DistMin", 0.1e-4)
        gmsh.model.mesh.field.setNumber(field_contact_thresh, "DistMax", 2e-4)
        fields.append(field_contact_thresh)
        print(f"  - Contact refinement: {self.mesh_size_contact * 1e7:.1f} nm")

        # Field for non-contact top surface
        field_surface_dist = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(field_surface_dist, "CurvesList", self.curves['top_surface'])
        field_surface_thresh = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(field_surface_thresh, "InField", field_surface_dist)
        gmsh.model.mesh.field.setNumber(field_surface_thresh, "SizeMin", self.mesh_size_surface)
        gmsh.model.mesh.field.setNumber(field_surface_thresh, "SizeMax", self.mesh_size_bulk)
        gmsh.model.mesh.field.setNumber(field_surface_thresh, "DistMin", 0.05e-4)
        gmsh.model.mesh.field.setNumber(field_surface_thresh, "DistMax", 1e-4)
        fields.append(field_surface_thresh)
        print(f"  - Surface refinement: {self.mesh_size_surface * 1e7:.1f} nm")

        # Field for p/p+ interface
        field_interface_dist = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(field_interface_dist, "CurvesList", [self.curves['interface_p_pplus']])
        field_interface_thresh = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(field_interface_thresh, "InField", field_interface_dist)
        gmsh.model.mesh.field.setNumber(field_interface_thresh, "SizeMin", self.mesh_size_interface)
        gmsh.model.mesh.field.setNumber(field_interface_thresh, "SizeMax", self.mesh_size_bulk)
        gmsh.model.mesh.field.setNumber(field_interface_thresh, "DistMin", 0.2e-4)
        gmsh.model.mesh.field.setNumber(field_interface_thresh, "DistMax", 4e-4)
        fields.append(field_interface_thresh)
        print(f"  - P/P+ interface refinement: {self.mesh_size_interface * 1e7:.1f} nm")

        # Final field: take the minimum of all defined fields
        field_min = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(field_min, "FieldsList", fields)
        gmsh.model.mesh.field.setAsBackgroundMesh(field_min)
        print(f"  - Bulk refinement: {self.mesh_size_bulk * 1e4:.1f} µm")

        # Use fields exclusively for mesh sizing
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    def _print_statistics(self):
        """Print mesh node/element count and quality metrics."""
        node_tags, _, _ = gmsh.model.mesh.getNodes()
        num_nodes = len(node_tags)
        num_elements = 0
        elem_types, elem_tags, _ = gmsh.model.mesh.getElements(2)
        for tags in elem_tags:
            num_elements += len(tags)

        print("\n" + "-" * 70)
        print("MESH STATISTICS")
        print("-" * 70)
        print(f"  Total nodes:     {num_nodes:,}")
        print(f"  Total elements:  {num_elements:,}")
        if num_elements > 0:
            qualities = gmsh.model.mesh.getElementQualities(elem_tags[0], "minSICN")
            print(f"  Min quality:     {np.min(qualities):.4f} (>0.3 is acceptable)")
            print(f"  Avg quality:     {np.mean(qualities):.4f}")
        print("-" * 70)

    def _print_physical_groups(self):
        print("\n" + "=" * 70)
        print("PHYSICAL GROUPS (queried from model)")
        print("=" * 70)
        for dim, dim_name in [(2, "Surfaces (regions)"), (1, "Curves (contacts/boundaries)")]:
            print(f"\n{dim_name}:")
            for tag in gmsh.model.getPhysicalGroups(dim):
                dim_tag, phys_tag = tag
                name = gmsh.model.getPhysicalName(dim_tag, phys_tag)
                ents = gmsh.model.getEntitiesForPhysicalGroup(dim_tag, phys_tag)
                n_ents = len(ents)
                print(f"  • {name}  [dim={dim_tag}, phys_tag={phys_tag}, entities={n_ents}]")
        print("=" * 70)


def main():
    """Example usage of the corrected PhotodiodeMesh class."""
    try:
        photodiode = PhotodiodeMesh(
            width=100.0,
            n_plus_thickness=0.5,
            p_thickness=50.0,
            p_plus_thickness=10.0,
            top_contact_x_start=45.0,
            top_contact_x_end=55.0,
            show_gui=True
        )
        photodiode.create_mesh()

    except Exception as e:
        print(f"\n✗ ERROR: An exception occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
