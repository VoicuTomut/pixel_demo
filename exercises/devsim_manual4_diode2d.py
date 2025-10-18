# Copyright 2013 DEVSIM LLC
#
# SPDX-License-Identifier: Apache-2.0

from devsim import set_parameter, solve

from devsim.python_packages.simple_physics import GetContactBiasName, PrintCurrents
import diode_common

from devsim import (
    add_1d_contact,
    add_1d_mesh_line,
    add_1d_region,
    add_2d_contact,
    add_2d_mesh_line,
    add_2d_region,
    add_gmsh_contact,
    add_gmsh_region,
    create_1d_mesh,
    create_2d_mesh,
    create_device,
    create_gmsh_mesh,
    finalize_mesh,
    get_contact_list,
    set_node_values,
    set_parameter,
)
# dio1
#
# Make doping a step function
# print dat to text file for viewing in grace
# verify currents analytically
# in dio2 add recombination
#

device = "MyDevice"
region = "MyRegion"

def Create2DMesh(device, region):
    create_2d_mesh(mesh="dio")
    add_2d_mesh_line(mesh="dio", dir="x", pos=0, ps=1e-6)
    add_2d_mesh_line(mesh="dio", dir="x", pos=0.5e-5, ps=1e-8)
    add_2d_mesh_line(mesh="dio", dir="x", pos=1e-5, ps=1e-6)
    add_2d_mesh_line(mesh="dio", dir="y", pos=0, ps=1e-6)
    add_2d_mesh_line(mesh="dio", dir="y", pos=1e-5, ps=1e-6)

    add_2d_mesh_line(mesh="dio", dir="x", pos=-1e-8, ps=1e-8)
    add_2d_mesh_line(mesh="dio", dir="x", pos=1.001e-5, ps=1e-8)

    add_2d_region(mesh="dio", material="Si", region=region)
    add_2d_region(mesh="dio", material="Si", region="air1", xl=-1e-8, xh=0)
    add_2d_region(mesh="dio", material="Si", region="air2", xl=1.0e-5, xh=1.001e-5)

    add_2d_contact(
        mesh="dio",
        name="top",
        material="metal",
        region=region,
        yl=0.8e-5,
        yh=1e-5,
        xl=0,
        xh=0,
        bloat=1e-10,
    )
    add_2d_contact(
        mesh="dio",
        name="bot",
        material="metal",
        region=region,
        xl=1e-5,
        xh=1e-5,
        bloat=1e-10,
    )

    finalize_mesh(mesh="dio")
    create_device(mesh="dio", device=device)



diode_common.SetParameters(device=device, region=region)

diode_common.SetNetDoping(device=device, region=region)

diode_common.InitialSolution(device, region)

# Initial DC solution
solve(type="dc", absolute_error=1.0, relative_error=1e-12, maximum_iterations=30)

diode_common.DriftDiffusionInitialSolution(device, region)
###
### Drift diffusion simulation at equilibrium
###
solve(type="dc", absolute_error=1e10, relative_error=1e-10, maximum_iterations=30)

####
#### Ramp the bias to 0.5 Volts
####
v = 0.0
while v < 0.51:
    set_parameter(device=device, name=GetContactBiasName("top"), value=v)
    solve(type="dc", absolute_error=1e10, relative_error=1e-10, maximum_iterations=30)
    PrintCurrents(device, "top")
    PrintCurrents(device, "bot")
    v += 0.1

val = 10
for i in range(2):
    set_parameter(device=device, name=GetContactBiasName("top"), value=val)
    data = solve(
        type="dc",
        absolute_error=1e10,
        relative_error=1e-10,
        maximum_iterations=30,
        info=True,
    )
    print(data["converged"])
    if not data["converged"]:
        val = 0.6

print(data)
for i in data["iterations"]:
    for d in i["devices"]:
        for r in d["regions"]:
            for e in r["equations"]:
                print(e)