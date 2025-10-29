from math import pi
import openmc
import openmc.deplete
import matplotlib.pyplot as plt
import numpy as np
import os
 
# Cross-section path
os.environ["OPENMC_CROSS_SECTIONS"] = "/home/emil/openmc/Cross_Section_Libraries/endfb-viii.0-hdf5/cross_sections.xml"
###############################################################################
#                              Define materials
###############################################################################

uo2 = openmc.Material(name='UO2 fuel at 2.4% wt enrichment')
uo2.set_density('g/cm3', 10.29769)
uo2.add_element('U', 1., enrichment=2.4)
uo2.add_element('O', 2.)


helium = openmc.Material(name='Helium for gap')
helium.set_density('g/cm3', 0.001598)
helium.add_element('He', 2.4044e-4)

zircaloy = openmc.Material(name='Zircaloy 4')
zircaloy.set_density('g/cm3', 6.55)
zircaloy.add_element('Sn', 0.014, 'wo')
zircaloy.add_element('Fe', 0.00165, 'wo')
zircaloy.add_element('Cr', 0.001, 'wo')
zircaloy.add_element('Zr', 0.98335, 'wo')

borated_water = openmc.Material(name='Borated water')
borated_water.set_density('g/cm3', 0.740582)
borated_water.add_element('B', 4.0e-5)
borated_water.add_element('H', 5.0e-2)
borated_water.add_element('O', 2.4e-2)
borated_water.add_s_alpha_beta('c_H_in_H2O')

###############################################################################
#                             Create geometry
###############################################################################

pitch = 1.25984
fuel_or = openmc.ZCylinder(r=0.39218, name='Fuel OR')
clad_ir = openmc.ZCylinder(r=0.40005, name='Clad IR')
clad_or = openmc.ZCylinder(r=0.45720, name='Clad OR')
box = openmc.model.RectangularPrism(pitch, pitch, boundary_type='reflective')

fuel = openmc.Cell(fill=uo2, region=-fuel_or)
gap = openmc.Cell(fill=helium, region=+fuel_or & -clad_ir)
clad = openmc.Cell(fill=zircaloy, region=+clad_ir & -clad_or)
water = openmc.Cell(fill=borated_water, region=+clad_or & -box)

geometry = openmc.Geometry([fuel, gap, clad, water])

uo2.volume = pi * fuel_or.r**2  # 2D → area

###############################################################################
#                     Transport calculation settings
###############################################################################

settings = openmc.Settings()
settings.batches = 50
settings.inactive = 10
settings.particles = 1000

bounds = [-0.62992, -0.62992, -1, 0.62992, 0.62992, 1]
uniform_dist = openmc.stats.Box(bounds[:3], bounds[3:])
settings.source = openmc.IndependentSource(
    space=uniform_dist, constraints={'fissionable': True})

entropy_mesh = openmc.RegularMesh()
entropy_mesh.lower_left = [-0.39218, -0.39218, -1.e50]
entropy_mesh.upper_right = [0.39218, 0.39218, 1.e50]
entropy_mesh.dimension = [10, 10, 1]
settings.entropy_mesh = entropy_mesh

###############################################################################
#                              Mesh tallies
###############################################################################

mesh = openmc.RegularMesh()
mesh.dimension = [100, 100]
mesh.lower_left = [-0.63, -0.63]
mesh.upper_right = [0.63, 0.63]
mesh_filter = openmc.MeshFilter(mesh)

# Define tallies
flux_tally = openmc.Tally(name='flux mesh tally')
flux_tally.filters = [mesh_filter]
flux_tally.scores = ['flux']

fission_tally = openmc.Tally(name='fission mesh tally')
fission_tally.filters = [mesh_filter]
fission_tally.scores = ['fission']

tallies = openmc.Tallies([flux_tally, fission_tally])

###############################################################################
#                   Initialize and run depletion calculation
###############################################################################

model = openmc.Model(geometry=geometry, settings=settings, tallies=tallies)
chain_file = 'chain_simple.xml'
op = openmc.deplete.CoupledOperator(model, chain_file)

#time_steps = [1.0, 1.0, 1.0, 1.0, 1.0]  # days
# cumulative_days = np.linspace(0.0, 360.0, 19)
# time_steps = np.diff(cumulative_days)
# burnup_cum = np.array([
#     0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
#     12.5, 15.0, 17.5, 20.0, 22.5, 25.0, 27.5, 30.0, 32.5, 35.0, 37.5,
#     40.0, 42.5, 45.0, 47.5, 50.0
# ])
burnup_cum = np.linspace(0.1, 50.0, 46)
burnup = np.diff(burnup_cum, prepend=0.0)
power = 174  # W/cm (for 2D simulation)
integrator = openmc.deplete.PredictorIntegrator(op, burnup, power, timestep_units='MWd/kg')

integrator.integrate()

###############################################################################
#                    Extract mesh results after each depletion step
###############################################################################

output_dir = "mesh_results"
os.makedirs(output_dir, exist_ok=True)

for step in range(len(burnup)):
    statepoint_file = f"statepoint.{(step+1):03d}.h5"
    
    if not os.path.exists(statepoint_file):
        print(f"{statepoint_file} not found, skipping.")
        continue

    print(f"Processing {statepoint_file} ...")
    sp = openmc.StatePoint(statepoint_file)

    # --- FLUX (neutron flux distribution) ---
    flux_tally = sp.get_tally(name='flux mesh tally')
    flux_data = flux_tally.get_reshaped_data()
    flux_data = np.array(flux_data).reshape(mesh.dimension)

    plt.figure(figsize=(6, 5))
    plt.imshow(flux_data, origin='lower', extent=[-0.63, 0.63, -0.63, 0.63])
    plt.title(f'Neutron Flux Distribution - Step {step+1}')
    plt.xlabel('x [cm]')
    plt.ylabel('y [cm]')
    plt.colorbar(label='Flux (a.u.)')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/flux_step{step+1}.png", dpi=300)
    plt.close()

    # --- FISSION → approximate HEAT distribution ---
    fission_tally = sp.get_tally(name='fission mesh tally')
    fission_data = fission_tally.get_reshaped_data()
    fission_data = np.array(fission_data).reshape(mesh.dimension)

    # Normalize to max for heatmap visualization
    heat_norm = fission_data / np.max(fission_data)

    plt.figure(figsize=(6, 5))
    plt.imshow(heat_norm, origin='lower', extent=[-0.63, 0.63, -0.63, 0.63], cmap='inferno')
    plt.title(f'Heat Distribution (Fission) - Step {step+1}')
    plt.xlabel('x [cm]')
    plt.ylabel('y [cm]')
    plt.colorbar(label='Relative Heat FISSION')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/heat_step{step+1}.png", dpi=300)
    plt.close()

print("\n All flux and fission distribution maps saved in:", output_dir)