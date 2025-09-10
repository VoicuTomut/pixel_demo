import devsim

def set_material_parameters(device, region, material_descriptor):
    """Sets the basic material parameters for Silicon."""
    devsim.set_parameter(device=device, region=region, name="Permittivity", value= material_descriptor["Permittivity"]["value"] )
    devsim.set_parameter(device=device, region=region, name="IntrinsicCarrierDensity", value=material_descriptor["IntrinsicCarrierDensity"]["value"])
    devsim.set_parameter(device=device, region=region, name="ElectronCharge", value=material_descriptor["ElectronCharge"]["value"])
    devsim.set_parameter(device=device, region=region, name="taun", value=material_descriptor["Tau"]["value"])
    devsim.set_parameter(device=device, region=region, name="taup", value=material_descriptor["Tau"]["value"])

