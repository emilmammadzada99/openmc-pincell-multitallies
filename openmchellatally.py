# ==============================================================
#  OpenMC Pincell Multi-Tallies Analysis and Visualization:
#  Neutron Spectrum Variations
#
#  Author      : Emil Mammadzada
#  Description : Multi-tally (mesh + energy) simulation, analysis,
#                and visualization of neutron spectrum variations
#                in a PWR fuel pin model using OpenMC.
#  Date        : October 2025
# ==============================================================
import os
import numpy as np
import openmc
import matplotlib.pyplot as plt

# Cross-section path
os.environ["OPENMC_CROSS_SECTIONS"] = "/home/emil/openmc/Cross_Section_Libraries/endfb-viii.0-hdf5/cross_sections.xml"

###############################################################################
# Materials
uo2 = openmc.Material(name='UO2 fuel at 2.4% wt enrichment')
uo2.set_density('g/cm3', 10.29769)
uo2.add_element('U', 1., enrichment=4.95) 
uo2.add_element('O', 2.)

helium = openmc.Material(name='Helium for gap')
helium.set_density('g/cm3', 0.001598)
helium.add_element('He', 1.0) 

zircaloy = openmc.Material(name='Zircaloy 4')
zircaloy.set_density('g/cm3', 6.55)
zircaloy.add_element('Sn', 0.014, 'wo')
zircaloy.add_element('Fe', 0.00165, 'wo')
zircaloy.add_element('Cr', 0.001, 'wo')
zircaloy.add_element('Zr', 0.98335, 'wo')

borated_water = openmc.Material(name='Borated water')
borated_water.set_density('g/cm3', 0.740582)
borated_water.add_element('B', 4.0e-5, 'ao') 
borated_water.add_element('H', 2.0/3.0, 'ao') 
borated_water.add_element('O', 1.0/3.0, 'ao') 
borated_water.add_s_alpha_beta('c_H_in_H2O')

materials = openmc.Materials([uo2, helium, zircaloy, borated_water])
materials.export_to_xml()

###############################################################################
# Geometry
fuel_or = openmc.ZCylinder(r=0.39218)
clad_ir = openmc.ZCylinder(r=0.40005)
clad_or = openmc.ZCylinder(r=0.45720)
pitch = 1.25984
box = openmc.model.RectangularPrism(pitch, pitch, boundary_type='reflective')

fuel = openmc.Cell(fill=uo2, region=-fuel_or)
gap = openmc.Cell(fill=helium, region=+fuel_or & -clad_ir)
clad = openmc.Cell(fill=zircaloy, region=+clad_ir & -clad_or)
water = openmc.Cell(fill=borated_water, region=+clad_or & -box)

geometry = openmc.Geometry([fuel, gap, clad, water])
geometry.export_to_xml()

###############################################################################
# Settings
settings = openmc.Settings()
settings.batches = 50
settings.particles = 1000

# Source
lower_left = (-pitch/2, -pitch/2, -1)
upper_right = (pitch/2, pitch/2, 1)
uniform_dist = openmc.stats.Box(lower_left, upper_right)
settings.source = openmc.IndependentSource(
    space=uniform_dist, constraints={'fissionable': True})

# Entropy mesh (optional)
entropy_mesh = openmc.RegularMesh()
entropy_mesh.lower_left = (-fuel_or.r, -fuel_or.r)
entropy_mesh.upper_right = (fuel_or.r, fuel_or.r)
entropy_mesh.dimension = (14, 14)
settings.entropy_mesh = entropy_mesh
settings.export_to_xml()

###############################################################################
# Mesh tanımlaması
mesh = openmc.RegularMesh()
mesh.dimension = (14, 14)
mesh.lower_left = (-pitch/2, -pitch/2)
mesh.upper_right = (pitch/2, pitch/2)
mesh_filter = openmc.MeshFilter(mesh)

# --- Reaction scores ekleniyor ---
reaction_scores = [
    'flux',
    'total',
    'fission',
    'absorption',
    '(n,elastic)',
    'scatter',
    '(n,2n)',
    '(n,3n)',
    '(n,gamma)',
    '(n,p)',
    '(n,a)',
    'nu-fission',
    'prompt-nu-fission'
]

# --- Mesh tally ---
mesh_tally = openmc.Tally(name="Mesh tally")
mesh_tally.filters = [mesh_filter]
mesh_tally.scores = reaction_scores

# --- Energy tally ---
energies = np.logspace(-5, np.log10(20e6), 501)
e_filter = openmc.EnergyFilter(energies)

energy_tally = openmc.Tally(name="Energy tally")
energy_tally.filters = [e_filter]
energy_tally.scores = reaction_scores

# --- Tallies export ---
tallies = openmc.Tallies([mesh_tally, energy_tally])
tallies.export_to_xml()

###############################################################################
model = openmc.Model(geometry=geometry, settings=settings, tallies=tallies)
sp_file = model.run(output=False)
with openmc.StatePoint(sp_file) as sp:
   
    energy_t = sp.get_tally(name="Energy tally")
    print("Available scores:", energy_t.scores)
    # energy_flux = energy_t.mean.flatten()
    # --- Energy tally ---
    energy_t = sp.get_tally(name="Energy tally")
    
    
    energy_flux        = energy_t.get_values(scores=['flux'])
    energy_total       = energy_t.get_values(scores=['total'])
    energy_fission     = energy_t.get_values(scores=['fission'])
    energy_absorption  = energy_t.get_values(scores=['absorption'])
    energy_n_elastic   = energy_t.get_values(scores=['(n,elastic)'])
    energy_scatter     = energy_t.get_values(scores=['scatter'])
    energy_n_2n        = energy_t.get_values(scores=['(n,2n)'])
    energy_n_3n        = energy_t.get_values(scores=['(n,3n)'])
    energy_n_gamma     = energy_t.get_values(scores=['(n,gamma)'])
    energy_n_p         = energy_t.get_values(scores=['(n,p)'])
    energy_n_a         = energy_t.get_values(scores=['(n,a)'])
    energy_nu_fission  = energy_t.get_values(scores=['nu-fission'])
    energy_prompt_nu_fission = energy_t.get_values(scores=['prompt-nu-fission'])
        
    
    energy_flux                 = np.squeeze(energy_flux)
    energy_total                = np.squeeze(energy_total)
    energy_fission              = np.squeeze(energy_fission)
    energy_absorption           = np.squeeze(energy_absorption)
    energy_n_elastic            = np.squeeze(energy_n_elastic)
    energy_scatter              = np.squeeze(energy_scatter)
    energy_n_2n                 = np.squeeze(energy_n_2n)
    energy_n_3n                 = np.squeeze(energy_n_3n)
    energy_n_gamma              = np.squeeze(energy_n_gamma)
    energy_n_p                  = np.squeeze(energy_n_p)
    energy_n_a                  = np.squeeze(energy_n_a)
    energy_nu_fission           = np.squeeze(energy_nu_fission)
    energy_prompt_nu_fission    = np.squeeze(energy_prompt_nu_fission)
        
    
    
    print(energy_flux)
    
    # --- Mesh tally ---
    mesh_t = sp.get_tally(name="Mesh tally")
    mesh_flux = mesh_t.mean.ravel()
    print(mesh_flux)
    
    
    mesh_flux = mesh_t.get_values(scores=['flux'])
    mesh_total = mesh_t.get_values(scores=['total'])
    mesh_fission = mesh_t.get_values(scores=['fission'])
    mesh_absorption = mesh_t.get_values(scores=['absorption'])
    mesh_elastic = mesh_t.get_values(scores=['(n,elastic)'])
    mesh_scatter = mesh_t.get_values(scores=['scatter'])
    mesh_n_2n = mesh_t.get_values(scores=['(n,2n)'])
    mesh_n_3n = mesh_t.get_values(scores=['(n,3n)'])
    mesh_n_gamma = mesh_t.get_values(scores=['(n,gamma)'])
    mesh_n_p = mesh_t.get_values(scores=['(n,p)'])
    mesh_n_a = mesh_t.get_values(scores=['(n,a)'])
    mesh_nu_fission = mesh_t.get_values(scores=['nu-fission'])
    mesh_prompt_nu_fission = mesh_t.get_values(scores=['prompt-nu-fission'])
    
    
    mesh_flux = np.squeeze(mesh_flux)
    mesh_total = np.squeeze(mesh_total)
    mesh_fission = np.squeeze(mesh_fission)
    mesh_absorption = np.squeeze(mesh_absorption)
    mesh_elastic = np.squeeze(mesh_elastic)
    mesh_scatter = np.squeeze(mesh_scatter)
    mesh_n_2n = np.squeeze(mesh_n_2n)
    mesh_n_3n = np.squeeze(mesh_n_3n)
    mesh_n_gamma = np.squeeze(mesh_n_gamma)
    mesh_n_p = np.squeeze(mesh_n_p)
    mesh_n_a = np.squeeze(mesh_n_a)
    mesh_nu_fission = np.squeeze(mesh_nu_fission)
    mesh_prompt_nu_fission = np.squeeze(mesh_prompt_nu_fission)
    
    
    mesh_flux_2d = mesh_flux.reshape(mesh.dimension)
    mesh_total_2d = mesh_total.reshape(mesh.dimension)
    mesh_fission_2d = mesh_fission.reshape(mesh.dimension)
    mesh_absorption_2d = mesh_absorption.reshape(mesh.dimension)
    mesh_elastic_2d = mesh_elastic.reshape(mesh.dimension)
    mesh_scatter_2d = mesh_scatter.reshape(mesh.dimension)
    mesh_n_2n_2d = mesh_n_2n.reshape(mesh.dimension)
    mesh_n_3n_2d = mesh_n_3n.reshape(mesh.dimension)
    mesh_n_gamma_2d = mesh_n_gamma.reshape(mesh.dimension)
    mesh_n_p_2d = mesh_n_p.reshape(mesh.dimension)
    mesh_n_a_2d = mesh_n_a.reshape(mesh.dimension)
    mesh_nu_fission_2d = mesh_nu_fission.reshape(mesh.dimension)
    mesh_prompt_nu_fission_2d = mesh_prompt_nu_fission.reshape(mesh.dimension)
        
    
   
    energy_diff = np.diff(energies)
    
    
    # flux_density: [flux_mean / (E_i+1 - E_i)]
    flux_density = energy_flux[:len(energy_diff)] / energy_diff           
    total_density = energy_total[:len(energy_diff)] / energy_diff
    fission_density = energy_fission[:len(energy_diff)] / energy_diff
    absorption_density = energy_absorption[:len(energy_diff)] / energy_diff
    elastic_density = energy_n_elastic[:len(energy_diff)] / energy_diff
    scatter_density = energy_scatter[:len(energy_diff)] / energy_diff
    n_2n_density = energy_n_2n[:len(energy_diff)] / energy_diff
    n_3n_density = energy_n_3n[:len(energy_diff)] / energy_diff
    n_gamma_density = energy_n_gamma[:len(energy_diff)] / energy_diff
    n_p_density = energy_n_p[:len(energy_diff)] / energy_diff
    n_a_density = energy_n_a[:len(energy_diff)] / energy_diff
    nu_fission_density = energy_nu_fission[:len(energy_diff)] / energy_diff
    prompt_nu_fission_density = energy_prompt_nu_fission[:len(energy_diff)] / energy_diff
        
    
    
    fig, ax = plt.subplots()
    ax.step(energies[:-1], flux_density, where='post', label='Flux')
    ax.step(energies[:-1], total_density, where='post', label='Total')
    ax.step(energies[:-1], fission_density, where='post', label='Fission')
    ax.step(energies[:-1], absorption_density, where='post', label='Absorption')
    ax.step(energies[:-1], elastic_density, where='post', label='(n,elastic)')
    ax.step(energies[:-1], scatter_density, where='post', label='Scatter')
    ax.step(energies[:-1], n_2n_density, where='post', label='(n,2n)')
    ax.step(energies[:-1], n_3n_density, where='post', label='(n,3n)')
    ax.step(energies[:-1], n_gamma_density, where='post', label='(n,gamma)')
    ax.step(energies[:-1], n_p_density, where='post', label='(n,p)')
    ax.step(energies[:-1], n_a_density, where='post', label='(n,a)')
    ax.step(energies[:-1], nu_fission_density, where='post', label='Nu-fission')
    ax.step(energies[:-1], prompt_nu_fission_density, where='post', label='Prompt-nu-fission')
    ax.set_xscale('log')
    ax.set_xlim(1e-4, 1e7)
    ax.set_yscale('log')
    ax.set_xlabel('Energy [eV]')
    ax.set_ylabel('Flux')
    ax.grid(True, which='both')
    ax.legend()
    plt.show()

    # --- Mesh plot ---
    x_coords = np.linspace(-pitch / 2, pitch / 2, mesh.dimension[0] + 1)
    y_coords = np.linspace(-pitch / 2, pitch / 2, mesh.dimension[1] + 1)
    # Mesh flux
    fig, ax = plt.subplots(figsize=(8, 6))
    # c = ax.pcolormesh(np.linspace(-pitch/2, pitch/2, mesh.dimension[0] + 1),
    #                   np.linspace(-pitch/2, pitch/2, mesh.dimension[1] + 1),
    #                   mesh_flux_2d, cmap='viridis', shading='auto')
    c = ax.pcolormesh(x_coords, y_coords, mesh_flux_2d, cmap='viridis', shading='auto')

   
    fig.colorbar(c, ax=ax, label='Flux')
    ax.set_xlabel('X [cm]')
    ax.set_ylabel('Y [cm]')
    ax.set_title('Mesh Flux Distribution')
    
    for i in range(mesh.dimension[0]):
        for j in range(mesh.dimension[1]):
            ax.text(x_coords[i] + (x_coords[i + 1] - x_coords[i]) / 2,
                    y_coords[j] + (y_coords[j + 1] - y_coords[j]) / 2,
                    f'({i},{j})', color='white', fontsize=8,
                    ha='center', va='center')
            
    
    ax.grid(True)
    plt.show()
    print("\nMesh Flux Distribution (i, j, Flux Value):")
    print("---------------------------------------------------")
    print(f"{'i':<5} {'j':<5} {'Flux Value':<20}")
    
    for i in range(mesh.dimension[0]):
        for j in range(mesh.dimension[1]):
            flux_value        = mesh_flux_2d[i, j]
            total_value       = mesh_total_2d[i, j]
            fission_value     = mesh_fission_2d[i, j]
            absorption_value  = mesh_absorption_2d[i, j]
            elastic_value     = mesh_elastic_2d[i, j]
            scatter_value     = mesh_scatter_2d[i, j]
            n_2n_value        = mesh_n_2n_2d[i, j]
            n_3n_value        = mesh_n_3n_2d[i, j]
            n_gamma_value     = mesh_n_gamma_2d[i, j]
            n_p_value         = mesh_n_p_2d[i, j]
            n_a_value         = mesh_n_a_2d[i, j]
            nu_fission_value  = mesh_nu_fission_2d[i, j]
            prompt_nu_fission_value = mesh_prompt_nu_fission_2d[i, j]
            # mesh_flux_2d = mesh_flux.reshape(mesh.dimension)
            # mesh_fission_2d = mesh_fission.reshape(mesh.dimension)
            # mesh_absorption_2d = mesh_absorption.reshape(mesh.dimension)
            # mesh_elastic_2d = mesh_elastic.reshape(mesh.dimension)
            # mesh_scatter_2d = mesh_scatter.reshape(mesh.dimension)
    
            normalized_flux_density = flux_density / flux_value
            normalized_total_density = total_density / total_value if total_value > 0 else 0
            normalized_fission_density = fission_density / fission_value if fission_value > 0 else 0
            normalized_absorption_density = absorption_density / absorption_value if absorption_value > 0 else 0
            normalized_elastic_density = elastic_density / elastic_value if elastic_value > 0 else 0
            normalized_scatter_density = scatter_density / scatter_value if scatter_value > 0 else 0
            normalized_n_2n_density = n_2n_density / n_2n_value if n_2n_value > 0 else 0
            normalized_n_3n_density = n_3n_density / n_3n_value if n_3n_value > 0 else 0
            normalized_n_gamma_density = n_gamma_density / n_gamma_value if n_gamma_value > 0 else 0
            normalized_n_p_density = n_p_density / n_p_value if n_p_value > 0 else 0
            normalized_n_a_density = n_a_density / n_a_value if n_a_value > 0 else 0
            normalized_nu_fission_density = nu_fission_density / nu_fission_value if nu_fission_value > 0 else 0
            normalized_prompt_nu_fission_density = prompt_nu_fission_density / prompt_nu_fission_value if prompt_nu_fission_value > 0 else 0
                        
            
            # if fission_value > 0:
            #     y_data = normalized_flux_density + normalized_fission_density
            # else:
            #     y_data = normalized_flux_density
            y_data = 0
            if flux_value > 0:
                y_data += normalized_flux_density
            if total_value > 0:
                y_data += normalized_total_density
            if fission_value > 0:
                y_data += normalized_fission_density
            if absorption_value > 0:
                y_data += normalized_absorption_density
            if elastic_value > 0:
                y_data += normalized_elastic_density
            if scatter_value > 0:
                y_data += normalized_scatter_density
            if n_2n_value > 0:
                y_data += normalized_n_2n_density
            if n_3n_value > 0:
                y_data += normalized_n_3n_density
            if n_gamma_value > 0:
                y_data += normalized_n_gamma_density
            if n_p_value > 0:
                y_data += normalized_n_p_density
            if n_a_value > 0:
                y_data += normalized_n_a_density
            if nu_fission_value > 0:
                y_data += normalized_nu_fission_density
            if prompt_nu_fission_value > 0:
                y_data += normalized_prompt_nu_fission_density
            # --- Tek figür içinde iki alt grafik ---
            fig = plt.figure(figsize=(10, 6))
    
           
            #ax1 = fig.add_axes([0.05, 0.35, 0.65, 0.6])  # [left, bottom, width, height]
            ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            ax1.step(energies[:-1], y_data, where='post')
            ax1.set_xscale('log')
            ax1.set_xlim(1e-3, 1e8)
            ax1.set_yscale('log')
            ax1.set_xlabel('Energy [eV]')
            ax1.set_ylabel('Normalized Flux')
            #ax1.grid(True, which='both')
    
            # 
            #ax2 = fig.add_axes([0.04, 0.35, 0.2, 0.3])  
            ax2 = fig.add_axes([0.15, 0.15, 0.25, 0.25])
            ax2.set_aspect('equal')
            ax2.set_xlim(-pitch / 2, pitch / 2)
            ax2.set_ylim(-pitch / 2, pitch / 2)
    
            # 
            fuel_circle = plt.Circle((0, 0), fuel_or.r, color='orange', alpha=0.6)
            gap_circle = plt.Circle((0, 0), clad_ir.r, color='lightgray', alpha=0.6)
            clad_circle = plt.Circle((0, 0), clad_or.r, color='gray', alpha=0.6)
            water_square = plt.Rectangle((-pitch/2, -pitch/2), pitch, pitch, fill=False, color='blue')
    
            ax2.add_patch(water_square)
            ax2.add_patch(clad_circle)
            ax2.add_patch(gap_circle)
            ax2.add_patch(fuel_circle)
    
            
            x_lines = np.linspace(-pitch/2, pitch/2, mesh.dimension[0]+1)
            y_lines = np.linspace(-pitch/2, pitch/2, mesh.dimension[1]+1)
            for x in x_lines:
                ax2.plot([x, x], [-pitch/2, pitch/2], color='black', lw=0.5, alpha=0.8)
            for y in y_lines:
                ax2.plot([-pitch/2, pitch/2], [y, y], color='black', lw=0.5, alpha=0.8)
    
            
            cx_sel = x_lines[i] + (x_lines[i+1] - x_lines[i])/2
            cy_sel = y_lines[j] + (y_lines[j+1] - y_lines[j])/2
            ax2.arrow(cx_sel, cy_sel+0.02, 0, -0.04, color='red', head_width=0.03, head_length=0.03, lw=1.5, length_includes_head=True)
            ax2.plot(cx_sel, cy_sel, 'ro', markersize=5)
    
            ax2.set_xticks([])
            ax2.set_yticks([])
            ax2.set_facecolor('#D6EAF8')
            filename = f"flux_cell_{i}_{j}.png"  
            plt.savefig(filename, dpi=300)        
            plt.close(fig)                  
    
            plt.show()