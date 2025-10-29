import openmc
import numpy as np
import matplotlib.pyplot as plt
import os

output_dir = "mesh_results1"
os.makedirs(output_dir, exist_ok=True)

statepoint_files = [f"openmc_simulation_n{i}.h5" for i in range(46)]

for step, sp_file in enumerate(statepoint_files):
    if not os.path.exists(sp_file):
        print(f"{sp_file} not found, skipping.")
        continue

    print(f"Processing {sp_file} ...")
    sp = openmc.StatePoint(sp_file)
    
    # --- Flux ---
    flux_tally = sp.get_tally(name='flux mesh tally')
    flux_mesh = flux_tally.filters[0].mesh
    nx, ny = flux_mesh.dimension
    flux_data = flux_tally.mean.reshape((nx, ny))

    # --- Fission / Heat ---
    fission_tally = sp.get_tally(name='fission mesh tally')
    fission_data = fission_tally.mean.reshape((nx, ny))
    
    V_MIN = 8.151502e-11
    V_MAX = 1.800931e-04

    # --- Tek Figür, İki Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Sol: Flux
    im1 = axes[0].imshow(flux_data, origin='lower',
                         extent=(flux_mesh.lower_left[0], flux_mesh.upper_right[0],
                                 flux_mesh.lower_left[1], flux_mesh.upper_right[1]))
    axes[0].set_title(f'Neutron Flux - Step {step}')
    axes[0].set_xlabel('x [cm]')
    axes[0].set_ylabel('y [cm]')
    cbar1 = plt.colorbar(im1, ax=axes[0])
    cbar1.set_label('Flux (a.u.)')

    # Sağ: Heat/Fission
    im2 = axes[1].imshow(fission_data, origin='lower',
                         extent=(flux_mesh.lower_left[0], flux_mesh.upper_right[0],
                                 flux_mesh.lower_left[1], flux_mesh.upper_right[1]),
                         cmap='hot', vmin=V_MIN, vmax=V_MAX)
    axes[1].set_title(f'Depletion - Step {step}')
    axes[1].set_xlabel('x [cm]')
    axes[1].set_ylabel('y [cm]')
    cbar2 = plt.colorbar(im2, ax=axes[1])
    cbar2.set_label('Relative Heat (Fission)')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/flux_heat_step{step}.png", dpi=300)
    plt.close(fig)

print("\n All combined flux + heat plots saved in", output_dir)