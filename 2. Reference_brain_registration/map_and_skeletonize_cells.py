
# This codes deforms segments to the reference brain
# Install environnement using conda env create --file map_and_skeletonize.yaml
# Version 0.1 02/05/2024 jbw

from pathlib import Path
import os
import sys
sys.path.append("/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/clem_zfish1")
from ANTs_registration_helpers_jbw import ANTsRegistrationHelpers

os.environ['ANTs_use_threads'] = "11"
os.environ['ANTs_bin_path'] = "/Users/jonathanboulanger-weill/Packages/install/bin"

ants_reg = ANTsRegistrationHelpers()

root_path = Path("/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/")
all_cells_path= Path("/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/clem_zfish1/traced_neurons/all_cells_111224/")

clem_fish1_cells = [entry.name for entry in Path(all_cells_path).iterdir() if entry.is_dir()]
clem_fish1_cells

for cell_name in clem_fish1_cells:
    ants_reg.convert_synapse_file(root_path=root_path / "clem_zfish1" / "traced_neurons" / "all_cells_111224",
                                  cell_name=cell_name,
                                  shift_x=0,
                                  shift_y=0,
                                  shift_z=0,
                                  scale_x=8,
                                  scale_y=8,
                                  scale_z=30,)

    ants_reg.map_and_skeletonize_cell(root_path=root_path / "clem_zfish1" / "traced_neurons" / "all_cells_111224",
                                      cell_name=cell_name,
                                      transformation_prefix_path=root_path / "clem_zfish1" / "transforms" / "clem_zfish1_to_zbrain_022824" / "ANTs_dfield",
                                      input_limit_x=523776,  # (1024 x pixel - 1) * 512 nm x-resolution
                                      input_limit_y=327168,  # (640 y pixel - 1) * 512 nm x-resolution
                                      input_limit_z=120000,  # (251 planes - 1) * 480 nm z-resolution
                                      input_scale_x=0.001,  # The lowres stack was reduced by factor 1000, so make it ym
                                      input_scale_y=0.001,
                                      input_scale_z=0.001)