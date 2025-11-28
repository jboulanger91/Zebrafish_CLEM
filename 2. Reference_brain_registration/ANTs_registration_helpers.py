import datetime
import os
import platform
import subprocess
import tempfile
from pathlib import Path
import nibabel as nib
import nrrd
import numpy as np
import pandas as pd
import pylab as pl
import skimage.metrics
import tifffile
from scipy.ndimage import zoom
import csv
import trimesh as tm
import navis
import skeletor as sk
import tomllib
import time

class ANTsRegistrationHelpers():
    def __init__(self, manual_opts_dict=None):

        self.opts_dict = {
            'interpolation_method': 'linear',
            'ANTs_verbose': 1,
            'tempdir': None,
            'num_cores': os.environ['ANTs_use_threads'],
            'ANTs_bin_path': os.environ['ANTs_bin_path'],
            'ANTs_use-histogram-matching': 0,
            'matching_metric': 'NMI',

            'rigid': {"use": True,
                      "t": "Rigid[0.1]",
                      "m": "MI[$1,$2,1,32,Regular,0.25]",  # $1, $2, source and target path
                      "c": "[1000x500x250x300,1e-8,10]",
                      "s": "3x2x1x0",
                      "f": "8x4x2x1"},

            'affine': {"use": True,
                       "t": "Affine[0.1]",
                       "m": "MI[$1,$2,1,32,Regular,0.25]",  # $1, $2, source and target path
                       "c": "[200x200x200x100,1e-8,10]",
                       "s": "3x2x1x0",
                       "f": "8x4x2x1"},

            'SyN': {"use": True,
                    "t": "SyN[0.1,6,0]",
                    "m": "CC[$1,$2,1,2]",  # $1, $2, source and target path
                    "c": "[200x200x200x100,1e-7,10]",
                    "s": "4x3x2x1",
                    "f": "12x8x4x2"},

            'BSplineSyn': {"use": False,
                           "t": "BSplineSyn[0.1,26,0,3]",
                           "m": "CC[$1,$2,1,4]",  # $1, $2, source and target path
                           "c": "[100x70x50x20,1e-7,10]",
                           "s": "3x2x1x0",
                           "f": "6x4x2x1"}
        }

        # Override some of the default values
        if manual_opts_dict is not None:
            for key in manual_opts_dict.keys():
                self.opts_dict[key] = manual_opts_dict[key]

    def convert_path_to_linux(self, path_name):

        if platform.system() == "Windows":
            path_name_linux = "/mnt/" + str(path_name)

            path_name_linux = path_name_linux.replace("\\", '/')
            path_name_linux = path_name_linux.replace(" ", '\\ ')

            path_name_linux = path_name_linux.replace("C:", 'c')
            path_name_linux = path_name_linux.replace("D:", 'd')
            path_name_linux = path_name_linux.replace("E:", 'e')
            path_name_linux = path_name_linux.replace("F:", 'f')
            path_name_linux = path_name_linux.replace("G:", 'g')
            path_name_linux = path_name_linux.replace("X:", 'x')
            path_name_linux = path_name_linux.replace("Y:", 'y')
            path_name_linux = path_name_linux.replace("Z:", 'z')
            path_name_linux = path_name_linux.replace("W:", 'w')
            path_name_linux = path_name_linux.replace("V:", 'v')

            return path_name_linux
        else:
            return str(path_name)

    def call_ANTs_command(self, command_list, stdin_file=None, stdout_file=None):

        if platform.system() in ["Linux", "Darwin"]:
            # If we are in linux or mac os, we can directly execute the commands
            subprocess.run(command_list,
                           stdin=open(stdin_file) if stdin_file is not None else None,
                           stdout=open(stdout_file, "w") if stdout_file is not None else None)

        elif platform.system() == "Windows":

            # in case of ants, we need to export the environmental variables for the ubuntu shell
            registration_commands = f"ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS={self.opts_dict['num_cores']}\n" \
                                    "export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS\n\n"

            registration_commands += ' '.join(command_list)

            if stdin_file is not None:
                registration_commands += f" < {stdin_file}"

            if stdout_file is not None:
                registration_commands += f" > {stdout_file}"

            registration_commands_path_temp = tempfile.NamedTemporaryFile(dir=self.opts_dict["tempdir"], suffix='.sh', delete=False)
            registration_commands_path_temp.close()

            registration_commands_path_linux = self.convert_path_to_linux(registration_commands_path_temp.name)

            file = open(registration_commands_path_temp.name, 'wb')
            file.write((registration_commands + '\n').encode())
            file.close()

            #####################
            print("Executing linux script inside windows shell....")
            print(registration_commands)

            result = subprocess.run(["bash", "-c", registration_commands_path_linux])

            if result.returncode != 0:
                result = subprocess.run(["ubuntu", "run", registration_commands_path_linux])
                print("UBTUNTUNUTNUNTUTN")

            os.remove(registration_commands_path_temp.name)

        # Sleep a bit, so the operating system may free the memory
        time.sleep(0.5)

    def ANTs_registration(self,
                          source_path,
                          target_path,
                          transformation_prefix_path,
                          manual_initial_moving_transform_path=None,
                          ANTs_dim=3):

        # Multi-channel registration allow for multiple sources and targets, usally one uses just a single one
        if not isinstance(source_path, list):
            source_path = [source_path]

        if not isinstance(target_path, list):
            target_path = [target_path]

        f_transformation_temp = tempfile.NamedTemporaryFile(dir=self.opts_dict["tempdir"], delete=False)
        f_transformation_temp.close()

        source_path_linux = [self.convert_path_to_linux(val) for val in source_path]
        target_path_linux = [self.convert_path_to_linux(val) for val in target_path]

        transformation_prefix_path_linux = self.convert_path_to_linux(transformation_prefix_path)
        f_transformation_temp_linux = self.convert_path_to_linux(f_transformation_temp.name)

        # Do the registration between source and reference
        registration_commands_list = [f"{self.opts_dict['ANTs_bin_path']}/antsRegistration",
                                      "-v", f"{self.opts_dict['ANTs_verbose']}",
                                      "-d", f"{ANTs_dim}",
                                      "--float", "1",
                                      "--winsorize-image-intensities", "[0.005, 0.995]",
                                      "--use-histogram-matching", f"{self.opts_dict['ANTs_use-histogram-matching']}",
                                      "-o", f_transformation_temp_linux]

        # In case, an initial moving transform was given, use this one, otherwise compute it (translations)
        if manual_initial_moving_transform_path is not None:
            registration_commands_list += ["--initial-moving-transform"]
            registration_commands_list += [self.convert_path_to_linux(manual_initial_moving_transform_path)]
        else:
            # # Only do one for the initial moving transform, take always the first from the list
            registration_commands_list += ["--initial-moving-transform"]
            registration_commands_list += [f"[{target_path_linux[0]},{source_path_linux[0]},1]"]

        # Build the registration steps
        for step_name in ["rigid", "affine", "SyN", "BSplineSyn"]:

            if self.opts_dict[step_name]["use"]:

                t = self.opts_dict[step_name]['t']
                m = self.opts_dict[step_name]['m']
                c = self.opts_dict[step_name]["c"]
                s = self.opts_dict[step_name]['s']
                f = self.opts_dict[step_name]['f']

                registration_commands_list += ["-t", t]

                # Allow for multi-modal registrations
                for i in range(len(target_path_linux)):
                    m_ = m.replace("$1", target_path_linux[i])
                    m_ = m_.replace("$2", source_path_linux[i])

                    registration_commands_list += ["-m", m_]

                registration_commands_list += ["-c", c,
                                               "-s", s,
                                               "-f", f]

        self.call_ANTs_command(registration_commands_list)

        # Concatenate all transformations into a proper matrix
        registration_commands_list = [f"{self.opts_dict['ANTs_bin_path']}/antsApplyTransforms",
                                      "--float",
                                      "-v", f"{self.opts_dict['ANTs_verbose']}",
                                      "-d", f"{ANTs_dim}",
                                      "-r", f"{target_path_linux[0]}",
                                      "-i", f"{source_path_linux[0]}",
                                      "-n", f"linear"]

        if self.opts_dict['SyN']["use"] or self.opts_dict['BSplineSyn']["use"]:
            registration_commands_list += ["--transform", f"{f_transformation_temp_linux}1Warp.nii.gz"]
        registration_commands_list += ["--transform", f"{f_transformation_temp_linux}0GenericAffine.mat"]

        registration_commands_list += ["-o", f"[{transformation_prefix_path_linux}.nii.gz,1]"]

        # RUN
        self.call_ANTs_command(registration_commands_list)

        # Do the same also for the inverse transformations
        registration_commands_list = [f"{self.opts_dict['ANTs_bin_path']}/antsApplyTransforms",
                                      "--float",
                                      "-v", f"{self.opts_dict['ANTs_verbose']}",
                                      "-d", f"{ANTs_dim}",
                                      "-r", f"{source_path_linux[0]}",
                                      "-i", f"{target_path_linux[0]}",
                                      "-n", f"linear"]

        registration_commands_list += ["--transform", f"[{f_transformation_temp_linux}0GenericAffine.mat,1]"]

        if self.opts_dict['SyN']["use"] or self.opts_dict['BSplineSyn']["use"]:
            registration_commands_list += ["--transform", f"{f_transformation_temp_linux}1InverseWarp.nii.gz"]

        registration_commands_list += ["-o", f"[{transformation_prefix_path_linux}_inverse.nii.gz,1]"]

        # RUN
        self.call_ANTs_command(registration_commands_list)

        # The individual transforms can already be deleted, we keep the concatenated one for later.
        os.remove(f"{f_transformation_temp.name}0GenericAffine.mat")

        if self.opts_dict['SyN']["use"] or self.opts_dict['BSplineSyn']["use"]:
            os.remove(f"{f_transformation_temp.name}1Warp.nii.gz")
            os.remove(f"{f_transformation_temp.name}1InverseWarp.nii.gz")

        os.remove(f_transformation_temp.name)

    def ANTs_applytransform(self,
                            source_path,
                            target_path,
                            output_path,
                            transformation_prefix_path,
                            use_forward_transformation=True,
                            ANTs_dim=3):

        source_path_linux = self.convert_path_to_linux(source_path)
        target_path_linux = self.convert_path_to_linux(target_path)
        output_path_linux = self.convert_path_to_linux(output_path)
        transformation_prefix_path_linux = self.convert_path_to_linux(transformation_prefix_path)

        # NOTE that concatenation of transforms not always works reliably, so do multiple transforms sequentially

        # Apply the concatenated transform to our source, transforming it onto the target
        registration_commands_list = [f"{self.opts_dict['ANTs_bin_path']}/antsApplyTransforms",
                                      "--float",
                                      "-v", f"{self.opts_dict['ANTs_verbose']}",
                                      "-d", f"{ANTs_dim}",
                                      "-i", f"{source_path_linux}",
                                      "-r", f"{target_path_linux}",
                                      "-n", self.opts_dict['interpolation_method']]

        if use_forward_transformation:
            registration_commands_list += ["--transform", f"{transformation_prefix_path_linux}.nii.gz"]
        else:
            registration_commands_list += ["--transform", f"{transformation_prefix_path_linux}_inverse.nii.gz"]

        registration_commands_list += ["-o", output_path_linux]

        # RUN
        self.call_ANTs_command(registration_commands_list)

        # ants makes it a 32bit float, convert it back to 16bit uint
        readdata, header = nrrd.read(str(output_path))
        readdata = readdata.astype(np.uint16)
        header["encoding"] = 'gzip'
        header["type"] = 'uint16'
        nrrd.write(str(output_path), readdata, header)

    def ANTs_applytransform_to_points(self,
                                      data_points,
                                      transformation_prefix_path,
                                      use_forward_transformation=True,
                                      ANTs_dim=3,
                                      input_limit_x=None,
                                      input_limit_y=None,
                                      input_limit_z=None,
                                      input_shift_x=0, input_scale_x=1,
                                      input_shift_y=0, input_scale_y=1,
                                      input_shift_z=0, input_scale_z=1,
                                      input_swap_xy=False,
                                      output_shift_x=0, output_scale_x=1,
                                      output_shift_y=0, output_scale_y=1,
                                      output_shift_z=0, output_scale_z=1,
                                      output_swap_xy=False):

        all_data_points_path = tempfile.NamedTemporaryFile(dir=self.opts_dict["tempdir"], suffix='.csv', delete=False)
        all_data_points_path.close()

        all_data_points_registered_path = tempfile.NamedTemporaryFile(dir=self.opts_dict["tempdir"], suffix='.csv', delete=False)
        all_data_points_registered_path.close()

        all_data_points_path_linux = self.convert_path_to_linux(all_data_points_path.name)
        all_data_points_registered_linux = self.convert_path_to_linux(all_data_points_registered_path.name)
        transformation_prefix_path_linux = self.convert_path_to_linux(transformation_prefix_path)

        # ANTs wants four columns
        data_points_ants = np.c_[data_points, np.zeros(data_points.shape[0])]

        # Limit because ANTs makes mistakes if points are outside the orignal volume
        if input_limit_x is not None:
            data_points_ants[data_points_ants[:, 0] > input_limit_x, 0] = input_limit_x

        if input_limit_y is not None:
            data_points_ants[data_points_ants[:, 1] > input_limit_y, 1] = input_limit_y

        if input_limit_z is not None:
            data_points_ants[data_points_ants[:, 2] > input_limit_z, 2] = input_limit_z

        # Basic transformation
        data_points_ants[:, 0] = input_shift_x + input_scale_x * data_points_ants[:, 0]
        data_points_ants[:, 1] = input_shift_y + input_scale_y * data_points_ants[:, 1]
        data_points_ants[:, 2] = input_shift_z + input_scale_z * data_points_ants[:, 2]

        if input_swap_xy:
            data_points_ants = data_points_ants[:, [1, 0, 2]]  # Swap x and y

        np.savetxt(all_data_points_path.name, data_points_ants, delimiter=',', header="x,y,z,t", comments='')

        registration_commands_list = [f"{self.opts_dict['ANTs_bin_path']}/antsApplyTransformsToPoints",
                                      "--precision", "0",  # float32 is enough
                                      "--dimensionality", f"{ANTs_dim}",
                                      "--input", f"{all_data_points_path_linux}",
                                      "--output", f"{all_data_points_registered_linux}"]

        # When mapping points, the forward transformation is actually the inverse
        if use_forward_transformation:
            registration_commands_list += ["--transform", f"{transformation_prefix_path_linux}_inverse.nii.gz"]
        else:
            registration_commands_list += ["--transform", f"{transformation_prefix_path_linux}.nii.gz"]

        registration_commands_list += ["-o", all_data_points_registered_linux]

        self.call_ANTs_command(registration_commands_list)

        # Make sure when loading, the data is 2D, even if it is a single data line
        transformed_points = np.loadtxt(all_data_points_registered_path.name, delimiter=',', skiprows=1, usecols=(0, 1, 2), ndmin=2)

        if output_swap_xy:
            transformed_points = transformed_points[:, [1, 0, 2]]  # Swap x and y

        # Basic transformation
        transformed_points[:, 0] = output_shift_x + output_scale_x * transformed_points[:, 0]
        transformed_points[:, 1] = output_shift_y + output_scale_y * transformed_points[:, 1]
        transformed_points[:, 2] = output_shift_z + output_scale_z * transformed_points[:, 2]

        os.remove(all_data_points_path.name)
        os.remove(all_data_points_registered_path.name)

        return transformed_points

    def ANTS_applytransform_to_obj(self,
                                   input_filename,
                                   output_filename,
                                   transformation_prefix_path,
                                   use_forward_transformation=True,
                                   input_limit_x=None,
                                   input_limit_y=None,
                                   input_limit_z=None,
                                   input_shift_x=0, input_scale_x=1,
                                   input_shift_y=0, input_scale_y=1,
                                   input_shift_z=0, input_scale_z=1,
                                   input_swap_xy=False,
                                   output_shift_x=0, output_scale_x=1,
                                   output_shift_y=0, output_scale_y=1,
                                   output_shift_z=0, output_scale_z=1,
                                   output_swap_xy=False):

        # Use trimesh to load vertices and faces
        mesh = tm.load(input_filename)

        mesh.vertices = self.ANTs_applytransform_to_points(mesh.vertices,
                                                           transformation_prefix_path,
                                                           use_forward_transformation=use_forward_transformation,
                                                           ANTs_dim=3,
                                                           input_limit_x=input_limit_x,
                                                           input_limit_y=input_limit_y,
                                                           input_limit_z=input_limit_z,
                                                           input_shift_x=input_shift_x, input_scale_x=input_scale_x,
                                                           input_shift_y=input_shift_y, input_scale_y=input_scale_y,
                                                           input_shift_z=input_shift_z, input_scale_z=input_scale_z,
                                                           input_swap_xy=input_swap_xy,
                                                           output_shift_x=output_shift_x, output_scale_x=output_scale_x,
                                                           output_shift_y=output_shift_y, output_scale_y=output_scale_y,
                                                           output_shift_z=output_shift_z, output_scale_z=output_scale_z,
                                                           output_swap_xy=output_swap_xy)

        mesh.export(output_filename)

    def ANTs_applytransform_to_swc(self,
                                   input_filename,
                                   output_filename,
                                   transformation_prefix_path,
                                   use_forward_transformation=True,
                                   node_size_scale=1,
                                   input_skiprows=0,
                                   input_limit_x=None,
                                   input_limit_y=None,
                                   input_limit_z=None,
                                   input_shift_x=0, input_scale_x=1,
                                   input_shift_y=0, input_scale_y=1,
                                   input_shift_z=0, input_scale_z=1,
                                   input_swap_xy=False,
                                   output_shift_x=0, output_scale_x=1,
                                   output_shift_y=0, output_scale_y=1,
                                   output_shift_z=0, output_scale_z=1,
                                   output_swap_xy=False):

        # Repair some strange swc coming from SNT
        f_swc = open(input_filename, "r")
        temp = f_swc.read()
        f_swc.close()

        temp = temp.replace("\t", " ")

        f_swc_temp = tempfile.NamedTemporaryFile(mode='w', dir=self.opts_dict["tempdir"], delete=False)
        f_swc_temp.write(temp)
        f_swc_temp.close()

        cell_data = np.loadtxt(f_swc_temp.name, skiprows=input_skiprows, delimiter=' ', ndmin=2)

        os.remove(f_swc_temp.name)

        data_points = np.c_[cell_data[:, 2],
                            cell_data[:, 3],
                            cell_data[:, 4]]

        data_points_transformed = self.ANTs_applytransform_to_points(data_points,
                                                                     transformation_prefix_path,
                                                                     use_forward_transformation=use_forward_transformation,
                                                                     ANTs_dim=3,
                                                                     input_limit_x=input_limit_x,
                                                                     input_limit_y=input_limit_y,
                                                                     input_limit_z=input_limit_z,
                                                                     input_shift_x=input_shift_x, input_scale_x=input_scale_x,
                                                                     input_shift_y=input_shift_y, input_scale_y=input_scale_y,
                                                                     input_shift_z=input_shift_z, input_scale_z=input_scale_z,
                                                                     input_swap_xy=input_swap_xy,
                                                                     output_shift_x=output_shift_x, output_scale_x=output_scale_x,
                                                                     output_shift_y=output_shift_y, output_scale_y=output_scale_y,
                                                                     output_shift_z=output_shift_z, output_scale_z=output_scale_z,
                                                                     output_swap_xy=output_swap_xy)

        # the x_scale is computes as the ratio of target and source resolution
        transformed_cell_data = np.c_[cell_data[:, 0],
                                      cell_data[:, 1],
                                      data_points_transformed[:, 0],
                                      data_points_transformed[:, 1],
                                      data_points_transformed[:, 2],
                                      cell_data[:, 5] * node_size_scale,
                                      cell_data[:, 6]]

        df = pd.DataFrame(transformed_cell_data, columns=['index', 'neuronname', 'x', 'y', 'z', 'size', 'connects'])

        df["index"] = df["index"].astype(np.int64)
        df["neuronname"] = df["neuronname"].astype(np.int64)
        df["connects"] = df["connects"].astype(np.int64)

        df.to_csv(output_filename, index=False, sep=' ', header=False)

    def draw_swc_in_empty_volume(self, path_swc,target_path= None,
                                  target_x_size=621, target_y_size=1406, target_z_size=138,
                                  target_dx=0.798, target_dy=0.798, target_dz=2):

        path = Path(path_swc)
        output_path = Path(path).with_suffix(".nrrd")

        print(datetime.datetime.now(), "Running draw_cell_in_empty_volume.", locals())

        data = np.loadtxt(path, delimiter=' ', ndmin=2)

        #cell_body_locations = []
        stack = np.zeros((target_z_size * 2, target_y_size * 2, target_x_size * 2), dtype=int)

        for i in range(data.shape[0]):

            node_x = data[i, 2] / target_dx
            node_y = data[i, 3] / target_dy
            node_z = data[i, 4] / target_dz

            stack[int(round(node_z * 2)), int(round(node_y * 2)), int(round(node_x * 2))] = 1000

            parent_i = int(data[i, -1])
            if parent_i == -1:
                #cell_body_locations.append([node_x, node_y, node_z])
                continue

            j = np.where(data[:, 0] == parent_i)[0]

            if len(j) == 0:
                continue

            parent_x = data[j[0], 2] / target_dx
            parent_y = data[j[0], 3] / target_dy
            parent_z = data[j[0], 4] / target_dz

            dx = (node_x - parent_x) / 1000
            dy = (node_y - parent_y) / 1000
            dz = (node_z - parent_z) / 1000

            xs = np.round((parent_x + np.cumsum(np.ones(1000) * dx)) * 2).astype(np.int64)
            ys = np.round((parent_y + np.cumsum(np.ones(1000) * dy)) * 2).astype(np.int64)
            zs = np.round((parent_z + np.cumsum(np.ones(1000) * dz)) * 2).astype(np.int64)

            for k in range(1000):
                stack[zs[k], ys[k], xs[k]] = 1000

        # Draw the soma
        # Soma location in ym
        soma_x = data[0, 2] / target_dx * 2
        soma_y = data[0, 3] / target_dy * 2
        soma_z = data[0, 4] / target_dz * 2

        # Make a meshgrid for the volume
        x = np.arange(0, target_x_size*2)
        y = np.arange(0, target_y_size*2)
        z = np.arange(0, target_z_size*2)

        # Strangely, the order of the meshgrid needs to be y z x
        YY, ZZ, XX = np.meshgrid(y, z, x)
        soma_r = 4
        # Select a radius region around the soma
        ind = np.sqrt(((ZZ - soma_z)) ** 2 + ((YY - soma_y)) ** 2 + ((XX - soma_x)) ** 2) < soma_r

        stack[ind] = 1000

        stack = zoom(stack, 0.5)

        stack[stack > 20] = 255
        stack[stack <= 20] = 0

        stack = stack.astype(np.uint16)
        header = {'type': 'uint16',
                  'encoding': 'raw',
                  'endian': 'big',
                  'dimension': 3,
                  'sizes': stack.shape,
                  'space dimension': 3,
                  'space directions': [[target_dx, 0, 0], [0, target_dy, 0],
                                       [0, 0, target_dz]],
                  'space units': ['microns', 'microns', 'microns']}

        if target_path != None:
            output_path = target_path

        nrrd.write(str(output_path), stack, header, index_order='C')

    def ANTs_registration_planes_to_volume(self,
                                           planes_path,
                                           volume_path,
                                           transforms_prefix_path,
                                           show_mapping_quality=True):

        print('Mapping planes to a volume')

        planes_data, planes_header_3D = nrrd.read(str(planes_path))
        volume_data, volume_header_3D = nrrd.read(str(volume_path))

        dim_x_planes = planes_data.shape[0]
        dim_y_planes = planes_data.shape[1]

        if len(planes_data.shape) == 3:
            dim_z_planes = planes_data.shape[2]
        else:
            planes_data = planes_data[:,:,np.newaxis]
            dim_z_planes = planes_data.shape[2]
        dx_planes = planes_header_3D["space directions"][0, 0]
        dy_planes = planes_header_3D["space directions"][1, 1]
        dz_planes = planes_header_3D["space directions"][2, 2]

        dim_x_volume = volume_data.shape[0]
        dim_y_volume = volume_data.shape[1]
        dim_z_volume = volume_data.shape[2]
        dx_volume = volume_header_3D["space directions"][0, 0]
        dy_volume = volume_header_3D["space directions"][1, 1]
        dz_volume = volume_header_3D["space directions"][2, 2]

        planes_header_2D = {'type': 'uint16',
                           'encoding': 'raw',
                           'endian': 'big',
                           'dimension': 2,
                           'sizes': (dim_x_planes, dim_y_planes),
                           'space dimension': 2,
                           'space directions': [[dx_planes, 0], [0, dy_planes]],
                           'space units': ['microns', 'microns']}

        volume_header_2D = {'type': 'uint16',
                           'encoding': 'raw',
                           'endian': 'big',
                           'dimension': 2,
                           'sizes': (dim_x_volume, dim_y_volume),
                           'space dimension': 2,
                           'space directions': [[dx_volume, 0], [0, dy_volume]],
                           'space units': ['microns', 'microns']}

        # We start with an empty deformation field which has huge dx,dy,dz, so it would always pick from the void
        dfield_3D = np.zeros((dim_x_volume, dim_y_volume, 3, dim_z_volume, 1), dtype=np.float32) + 10000
        dfield_inverse_3D = np.zeros((dim_x_planes, dim_y_planes, 3, dim_z_planes, 1), dtype=np.float32) + 10000

        # Loop over all source z_planes
        for planes_z in range(dim_z_planes):

            all_matching_qualities = []
            all_transformation_prefices = []

            search_volume_zs = range(dim_z_volume)

            # make an empty rgb volume stack (cxyz)
            if show_mapping_quality:
                target_volume_rgb = np.array([volume_data, np.zeros_like(volume_data),np.zeros_like(volume_data)], dtype=np.uint16)

            for volume_z in search_volume_zs:
                # Perform 2D ants mapping of this plane to each
                if np.nansum(volume_data[:, :, volume_z]) == 0:
                    all_matching_qualities.append(np.inf)
                    all_transformation_prefices.append('empty_plane_in_volume')
                    continue

                # Save the source and target planes as temporary nrrd files
                f_planes_temp = tempfile.NamedTemporaryFile(dir=self.opts_dict['tempdir'], suffix='.nrrd', delete=False)
                f_planes_temp.close()

                f_volume_temp = tempfile.NamedTemporaryFile(dir=self.opts_dict['tempdir'], suffix='.nrrd', delete=False)
                f_volume_temp.close()

                f_transformation_prefix_path_temp = tempfile.NamedTemporaryFile(dir=self.opts_dict['tempdir'], suffix=f'{volume_z}', delete=False)
                f_transformation_prefix_path_temp.close()

                f_transformed_planes_temp = tempfile.NamedTemporaryFile(dir=self.opts_dict['tempdir'], suffix='.nrrd', delete=False)
                f_transformed_planes_temp.close()

                nrrd.write(f_planes_temp.name, planes_data[:, :, planes_z].astype(np.uint16), header=planes_header_2D)
                nrrd.write(f_volume_temp.name, volume_data[:, :, volume_z].astype(np.uint16), header=volume_header_2D)

                # Perform a 2D ANTs registration
                print('Starting Registration')
                self.ANTs_registration(source_path=f_planes_temp.name,
                                       target_path=f_volume_temp.name,
                                       transformation_prefix_path=f_transformation_prefix_path_temp.name,
                                       ANTs_dim=2)

                # Use the computed transformation to move the plane into the plane from the volume
                self.ANTs_applytransform(source_path=f_planes_temp.name,
                                         target_path=f_volume_temp.name,
                                         output_path=f_transformed_planes_temp.name,
                                         transformation_prefix_path=f_transformation_prefix_path_temp.name,
                                         use_forward_transformation=True,
                                         ANTs_dim=2)

                # Load the result
                moved_data, moved_header = nrrd.read(f_transformed_planes_temp.name)

                # Compute the matching quality between the two images, different metrics are possible
                if self.opts_dict['matching_metric'] == "NMI":
                    matching_quality = skimage.metrics.normalized_mutual_information(moved_data, volume_data[:, :, volume_z])
                elif self.opts_dict['matching_metric'] == "MSE":
                    matching_quality = skimage.metrics.mean_squared_error(moved_data, volume_data[:, :, volume_z])
                elif self.opts_dict['matching_metric'] == "SSIM":
                    matching_quality = skimage.metrics.structural_similarity(moved_data, volume_data[:, :, volume_z])
                else:
                    raise

                # Remember the quality and the transformation stacks
                all_matching_qualities.append(matching_quality)
                all_transformation_prefices.append(f_transformation_prefix_path_temp.name)

                # place the target moved data into the volume stack green channel
                if show_mapping_quality:
                    target_volume_rgb[1, :, :, volume_z] = moved_data

                # Delete temporary files
                os.remove(f_planes_temp.name)
                os.remove(f_volume_temp.name)
                os.remove(f_transformed_planes_temp.name)
                os.remove(f_transformation_prefix_path_temp.name)

            if show_mapping_quality:
                pl.figure(1)
                pl.plot(search_volume_zs, all_matching_qualities, '-o', label=f'planes_z: {planes_z}')

            # Get the best target plane
            if self.opts_dict['matching_metric'] == "MSE":
                i = np.argmin(all_matching_qualities)
            else:
                i = np.argmax(all_matching_qualities)

            best_volume_z = search_volume_zs[i]
            best_matching_quality = all_matching_qualities[i]
            best_transformation_prefix = all_transformation_prefices[i]

            print("Best!", best_transformation_prefix)

            # Load the temporary nii.gz files and bring all dfields it into the XYCZT order
            dfield_2D = nib.load(str(best_transformation_prefix) + ".nii.gz")
            dfield_2D = np.array(dfield_2D.dataobj, dtype=np.float32)
            dfield_2D = dfield_2D.transpose([0, 1, 4, 2, 3])

            dfield_inverse_2D = nib.load(str(best_transformation_prefix) + "_inverse.nii.gz")
            dfield_inverse_2D = np.array(dfield_inverse_2D.dataobj, dtype=np.float32)
            dfield_inverse_2D = dfield_inverse_2D.transpose([0, 1, 4, 2, 3])

            # The first and second column represent pixel position in target space
            # The third column is the color channel, representating, dx, dy, dz
            # dx, dy, dz mean how much one needs to move from the target location to grab the pixel brightness from the source

            # The dx dy warp fields remain as found but the 2D registration
            dfield_3D[:, :, 0, best_volume_z, 0] = dfield_2D[:, :, 0, 0, 0]
            dfield_3D[:, :, 1, best_volume_z, 0] = dfield_2D[:, :, 1, 0, 0]

            # dz need to be scaled by the respective coordinate systems
            dfield_3D[:, :, 2, best_volume_z, 0] = -best_volume_z * dz_volume + planes_z * dz_planes

            # Do the same for the inverse
            dfield_inverse_3D[:, :, 0, planes_z, 0] = dfield_inverse_2D[:, :, 0, 0, 0]
            dfield_inverse_3D[:, :, 1, planes_z, 0] = dfield_inverse_2D[:, :, 1, 0, 0]
            dfield_inverse_3D[:, :, 2, planes_z, 0] = -planes_z * dz_planes + best_volume_z * dz_volume

            # Delete all temporary 2D warp fields
            for transformation_prefix in all_transformation_prefices:
                if 'empty_plane_in_volume' not in transformation_prefix:
                    os.remove(str(transformation_prefix) + ".nii.gz")
                    os.remove(str(transformation_prefix) + "_inverse.nii.gz")

            # need to transpoose cxyz -> zyxc
            if show_mapping_quality:
                tifffile.imwrite(str(transforms_prefix_path) + f'_mapping_result_source_plane{planes_z}.tiff',
                                 target_volume_rgb.transpose([3, 2, 1, 0]), photometric='rgb')

        if show_mapping_quality:
            pl.figure(1)
            pl.ylabel("Matching metric")
            pl.xlabel("Volume stack z")
            pl.legend()
            pl.savefig(str(transforms_prefix_path) + f"_mapping_matching_metric.png")
            pl.close()

        ##################
        # Save the transform to a compressed nii file

        # Transform the dfield to XYZTC, which the nifti libray needs to store the file
        dfield_3D = dfield_3D.transpose([0, 1, 3, 4, 2])

        # Make a nifti file with correct target resolution parameters
        dfield_3D = nib.Nifti1Image(dfield_3D, affine=np.array([[-dx_volume, 0., 0., -0.],
                                                                [0., -dy_volume, 0., -0.],
                                                                [0., 0., dz_volume, 0.],
                                                                [0., 0., 0., 1.]]))

        dfield_3D.header["srow_x"] = [-dx_volume, 0., 0., -0.]
        dfield_3D.header["srow_y"] = [0., -dy_volume, 0., -0.]
        dfield_3D.header["srow_z"] = [0., 0., dz_volume, 0.]
        dfield_3D.header["pixdim"] = [1., dx_volume, dy_volume, dz_volume, 0., 0., 0., 0.]
        dfield_3D.header["regular"] = b'r'
        dfield_3D.header["intent_code"] = 1007
        dfield_3D.header["qform_code"] = 1
        dfield_3D.header["sform_code"] = 1
        dfield_3D.header["xyzt_units"] = 2
        dfield_3D.header["quatern_d"] = 1

        nib.save(dfield_3D, str(transforms_prefix_path) + ".nii.gz")

        ##################
        # Do the same for the inverse
        dfield_inverse_3D = dfield_inverse_3D.transpose([0, 1, 3, 4, 2])
        dfield_inverse_3D = nib.Nifti1Image(dfield_inverse_3D, affine=np.array([[-dx_planes, 0., 0., -0.],
                                                                                [0., -dy_planes, 0., -0.],
                                                                                [0., 0., dz_planes, 0.],
                                                                                [0., 0., 0., 1.]]))

        dfield_inverse_3D.header["srow_x"] = [-dx_planes, 0., 0., 0.]
        dfield_inverse_3D.header["srow_y"] = [0., -dy_planes, 0., 0.]
        dfield_inverse_3D.header["srow_z"] = [0., 0., dz_planes, 0.]
        dfield_inverse_3D.header["pixdim"] = [1., dx_planes, dy_planes, dz_planes, 0., 0., 0., 0.]
        dfield_inverse_3D.header["regular"] = b'r'
        dfield_inverse_3D.header["intent_code"] = 1007
        dfield_inverse_3D.header["qform_code"] = 1
        dfield_inverse_3D.header["sform_code"] = 1
        dfield_inverse_3D.header["xyzt_units"] = 2
        dfield_inverse_3D.header["quatern_d"] = 1

        nib.save(dfield_inverse_3D, str(transforms_prefix_path) + "_inverse.nii.gz")

    def invert_bigwarp_landmarks(self, input_path, output_path):
        """
        Just changes the columns in the bigwarp matrix, such that source becomes target.
        The csv mapping file must exist, with a suffix _bigwarp_landmarks.csv
        """

        landmarks = pd.read_csv(input_path, names=["a", 'b', 'c', 'd', 'e', 'f', 'g', 'h'])

        landmarks_inverted = landmarks.copy()

        landmarks_inverted["c"] = landmarks["f"]
        landmarks_inverted["d"] = landmarks["g"]
        landmarks_inverted["e"] = landmarks["h"]
        landmarks_inverted["f"] = landmarks["c"]
        landmarks_inverted["g"] = landmarks["d"]
        landmarks_inverted["h"] = landmarks["e"]

        landmarks_inverted.to_csv(output_path, sep=',', quoting=csv.QUOTE_ALL, index=False, header=False)

        print("Inverted points generated, please use bigwarp again to generate the transform dfield. Save with suffix '_bigwarp_dfield_inverse.tif'")

    def convert_bigwarp_dfield_to_ANTs_dfield(self,
                                              dx,
                                              dy,
                                              dz,
                                              bigwarp_dfield_path,
                                              ANTs_dfield_path):

        bigwarp_dfield = tifffile.imread(str(bigwarp_dfield_path))

        dfield_x = bigwarp_dfield[:, 0]
        dfield_y = bigwarp_dfield[:, 1]
        dfield_z = bigwarp_dfield[:, 2]

        ants_dfield = np.array([[dfield_x,
                                 dfield_y,
                                 dfield_z]]).transpose(4, 3, 2, 0, 1)

        # float 32 is precise enough and makes the file much smaller
        ants_dfield = ants_dfield.astype(np.float32)

        ants_dfield = nib.Nifti1Image(ants_dfield, affine=np.array([[-dx, 0., 0., -0.],
                                                                    [0., -dy, 0., -0.],
                                                                    [0., 0., dz, 0.],
                                                                    [0., 0., 0., 1.]]))

        ants_dfield.header["srow_x"] = [0., 0., 0., 0.]
        ants_dfield.header["srow_y"] = [0., 0., 0., 0.]
        ants_dfield.header["srow_z"] = [0., 0., 0., 0.]
        ants_dfield.header["pixdim"] = [1., dx, dy, dz, 0., 0., 0., 0.]
        ants_dfield.header["regular"] = b'r'
        ants_dfield.header["intent_code"] = 1007
        ants_dfield.header["qform_code"] = 1
        ants_dfield.header["sform_code"] = 0
        ants_dfield.header["xyzt_units"] = 2
        ants_dfield.header["quatern_d"] = 1

        nib.save(ants_dfield, str(ANTs_dfield_path))

    def map_and_skeletonize_cell(self,
                                 root_path,
                                 cell_name,
                                 transformation_prefix_path,
                                 use_forward_transformation=True,
                                 input_limit_x=None,
                                 input_limit_y=None,
                                 input_limit_z=None,
                                 input_shift_x=0, input_scale_x=1,
                                 input_shift_y=0, input_scale_y=1,
                                 input_shift_z=0, input_scale_z=1,
                                 input_swap_xy=False,
                                 output_shift_x=0, output_scale_x=1,
                                 output_shift_y=0, output_scale_y=1,
                                 output_shift_z=0, output_scale_z=1,
                                 output_swap_xy=False):

        root_path = Path(root_path)

        # Load the meta data file
        with open(root_path / cell_name / f"{cell_name}_metadata.txt", mode="rb") as fp:
            metadata = tomllib.load(fp)

        print(f"Mapping and skeletonization of cell {cell_name}.")

        print("Meta data:", metadata)

        # Make a folder for the mapped file
        (root_path / cell_name / "mapped").mkdir(exist_ok=True)

        for synapse_type_str in ['presynapses', 'postsynapses']:

            if (root_path / cell_name / f"{cell_name}_{synapse_type_str}.csv").exists():
                print(f"Mapping {synapse_type_str}")

                df = pd.read_csv(root_path / cell_name / f"{cell_name}_{synapse_type_str}.csv", comment='#', sep=' ',
                                 header=None,
                                 names=["partner_cell_id", "x", "y", "z", "radius"])

                if len(df) > 0:

                    df.loc[:, "radius"] = df["radius"] * input_scale_x  # Scale the synapse radius with input_scale x

                    # Make it a numpy array for the mapping function
                    points = np.array(df[["x", "y", "z"]], dtype=np.float64)

                    points.shape = (-1, 3) # Make sure it has a 2D shape, also when using single data points

                    points_transformed = self.ANTs_applytransform_to_points(points,
                                                                            transformation_prefix_path,
                                                                            use_forward_transformation=use_forward_transformation,
                                                                            ANTs_dim=3,
                                                                            input_limit_x=input_limit_x,
                                                                            input_limit_y=input_limit_y,
                                                                            input_limit_z=input_limit_z,
                                                                            input_shift_x=input_shift_x, input_scale_x=input_scale_x,
                                                                            input_shift_y=input_shift_y, input_scale_y=input_scale_y,
                                                                            input_shift_z=input_shift_z, input_scale_z=input_scale_z,
                                                                            input_swap_xy=input_swap_xy,
                                                                            output_shift_x=output_shift_x, output_scale_x=output_scale_x,
                                                                            output_shift_y=output_shift_y, output_scale_y=output_scale_y,
                                                                            output_shift_z=output_shift_z, output_scale_z=output_scale_z,
                                                                            output_swap_xy=output_swap_xy)

                    df_mapped = pd.DataFrame({'partner_cell_id': df['partner_cell_id'],
                                              'x': points_transformed[:, 0],
                                              'y': points_transformed[:, 1],
                                              'z': points_transformed[:, 2],
                                              'radius': df["radius"]})
                else:
                    df_mapped = df  # This will again store an empty file

                # Save the mapped synapse locations
                df_mapped.to_csv(root_path / cell_name / "mapped" / f"{cell_name}_{synapse_type_str}_mapped.csv",
                                 index=False, sep=' ', header=None, float_format='%.8f')

            # Draw the synapses as small spheres in a new mesh file
            for mapped_i in [0, 1]:

                if mapped_i == 0:
                    suffix_str = ""
                    folder_str = ""
                else:
                    suffix_str = "_mapped"
                    folder_str = "mapped"

                for synapse_type_str in ['presynapses', 'postsynapses']:
                    if (root_path / cell_name / folder_str / f"{cell_name}_{synapse_type_str}{suffix_str}.csv").exists():
                        df = pd.read_csv(root_path / cell_name / folder_str / f"{cell_name}_{synapse_type_str}{suffix_str}.csv",
                                         comment='#', sep=' ', header=None, names=["synapse_id", "x", "y", "z", "radius"])

                        spheres = []

                        for _, row in df.iterrows():
                            sphere = tm.creation.icosphere(radius=row["radius"], subdivisions=2)
                            sphere.apply_translation((row["x"], row["y"], row["z"]))

                            spheres.append(sphere)

                        if len(spheres) > 0:
                            scene = tm.Scene(spheres)
                            scene.export(root_path / cell_name / folder_str / f"{cell_name}_{synapse_type_str}{suffix_str}.obj")

        ################
        meshes_original = dict({})
        meshes_mapped = dict({})

        # Map dendrite, axon, and soma
        for part_name in ["soma", "dendrite", "axon"]:

            print("Mapping mesh", part_name)

            # Use trimesh to load vertices and faces
            part_path=root_path / cell_name / f"{cell_name}_{part_name}.obj"

            if os.path.isfile(part_path): 
                mesh = tm.load(root_path / cell_name / f"{cell_name}_{part_name}.obj")
                meshes_original[part_name] = mesh.copy()

                # Note that this requires the original stack to be in nm.
                # Having it properly in ym is only needed for skeletonizing of the unmapped obj
                meshes_original[part_name].vertices *= 0.001 # make it ym

                mesh.vertices = self.ANTs_applytransform_to_points(mesh.vertices,
                                                                transformation_prefix_path,
                                                                use_forward_transformation=use_forward_transformation,
                                                                ANTs_dim=3,
                                                                input_limit_x=input_limit_x,
                                                                input_limit_y=input_limit_y,
                                                                input_limit_z=input_limit_z,
                                                                input_shift_x=input_shift_x, input_scale_x=input_scale_x,
                                                                input_shift_y=input_shift_y, input_scale_y=input_scale_y,
                                                                input_shift_z=input_shift_z, input_scale_z=input_scale_z,
                                                                input_swap_xy=input_swap_xy,
                                                                output_shift_x=output_shift_x, output_scale_x=output_scale_x,
                                                                output_shift_y=output_shift_y, output_scale_y=output_scale_y,
                                                                output_shift_z=output_shift_z, output_scale_z=output_scale_z,
                                                                output_swap_xy=output_swap_xy)

                # Fix problems after mapping and store in dictionary
                meshes_mapped[part_name] = sk.pre.fix_mesh(mesh, fix_normals=True, inplace=False)

                # Save slightly simplified mapped meshes
                sk.pre.simplify(meshes_mapped[part_name], 0.75).export(root_path / cell_name / "mapped" / f"{cell_name}_{part_name}_mapped.obj")

        for mapped_i in [0, 1]:

            neurite_radius = 0.5
            soma_radius = 2
            skeletonize_inv_dist = 1.5

            if mapped_i == 0:
                suffix_str = ""
                folder_str = ""
                meshes = meshes_original

            else:
                suffix_str = "_mapped"
                folder_str = "mapped"
                meshes = meshes_mapped

            # If all parts are there
            if "soma" in meshes_mapped.keys() and "dendrite" in meshes_mapped.keys() and "axon" in meshes_mapped.keys():
                print("Making SWC, based on soma, dendrite, and axon")

                if mapped_i == 0:
                    # Save the complete mapped neuron as a single .obj
                    mesh_axon = tm.load(root_path / cell_name / f"{cell_name}_axon.obj")
                    mesh_dendrite = tm.load(root_path / cell_name / f"{cell_name}_dendrite.obj")
                    mesh_soma = tm.load(root_path / cell_name / f"{cell_name}_soma.obj")

                    # Merge and save
                    mesh_complete = tm.util.concatenate([mesh_axon, mesh_dendrite, mesh_soma])
                    sk.pre.simplify(mesh_complete, 0.75).export(root_path / cell_name / f"{cell_name}.obj")
                else:
                    # Save the complete mapped neuron as a single .obj
                    mesh_axon = tm.load(root_path / cell_name / "mapped" / f"{cell_name}_axon_mapped.obj")
                    mesh_dendrite = tm.load(root_path / cell_name / "mapped" / f"{cell_name}_dendrite_mapped.obj")
                    mesh_soma = tm.load(root_path / cell_name / "mapped" / f"{cell_name}_soma_mapped.obj")

                    # Merge and save
                    mesh_complete = tm.util.concatenate([mesh_axon, mesh_dendrite, mesh_soma])
                    sk.pre.simplify(mesh_complete, 0.75).export(root_path / cell_name / "mapped" / f"{cell_name}_mapped.obj")

                # Combine meshes
                #mesh_axon_dendrite = meshes["axon"].union(meshes["dendrite"], engine='blender')  # Problem with new trimesh version
                mesh_axon_dendrite = tm.util.concatenate([meshes["axon"], meshes["dendrite"]])

                # Get the location of the soma by averaging soma obj vertices
                soma_x, soma_y, soma_z = np.mean(meshes["soma"].triangles_center, axis=0)

                # Skeletonize the axons and dendrites at 1.5 um precision
                skel = sk.skeletonize.by_teasar(mesh_axon_dendrite, inv_dist=skeletonize_inv_dist)

                # Remove some potential perpedicular branches that should not be there
                skel = sk.post.clean_up(skel)
                skel = skel.reindex()

                # Continue with the swc as a pandas dataframe
                df_swc = skel.swc

                # Make a standard axon/dendrite radius everywhere
                df_swc["radius"] = neurite_radius
                df_swc["label"] = 0

                # Repair gaps in the skeleton using navis
                x = navis.read_swc(df_swc)
                x = navis.heal_skeleton(x, method='ALL', max_dist=None, min_size=None, drop_disc=False, mask=None, inplace=False)

                # Save as a temporary swc
                f_swc_temp = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.swc')
                f_swc_temp.close()

                x.to_swc(f_swc_temp.name)

                # Read it as a dataframe for further processing
                df_swc = pd.read_csv(f_swc_temp.name, sep=" ", names=["node_id", 'label', 'x', 'y', 'z', 'radius', 'parent_id'], comment='#', header=None)

                # Delete temporary file
                os.remove(f_swc_temp.name)

                # Add the soma. For this need to move parent ids and node ids
                df_swc.loc[:, "node_id"] += 1
                df_swc.loc[df_swc["parent_id"] > -1, "parent_id"] += 1

                # Find the row that is closest to the soma
                i_min = ((df_swc["x"] - soma_x) ** 2 + (df_swc["y"] - soma_y) ** 2 + (df_swc["z"] - soma_z) ** 2).argmin()

                # If that node does not have a parent, then set the new soma as the parent
                if df_swc.loc[i_min, "parent_id"] == -1:
                    df_swc.loc[i_min, "parent_id"] = 0

                    # Soma always has node_id = 1
                    soma_row = pd.DataFrame({"node_id": 1, "label": 1, "x": soma_x, "y": soma_y, "z": soma_z, "radius": soma_radius, "parent_id": -1}, index=[0])
                else:
                    # Otherwise make the soma the child of that node
                    node_id = df_swc.loc[i_min, "node_id"]

                    # Soma always has node_id = 1
                    soma_row = pd.DataFrame({"node_id": 1, "label": 1, "x": soma_x, "y": soma_y, "z": soma_z, "radius": soma_radius, "parent_id": node_id}, index=[0])

                # Add soma to the beginning, make sure this does not produce double indices
                df_swc = pd.concat([soma_row, df_swc], ignore_index=True)

                # Label what is dendrite or axon, bases on minimal distances to the provided meshes
                for i, row in df_swc.iterrows():

                    # Ignore the soma, it is already labeled with = 1
                    if row["node_id"] == 1:
                        continue

                    d_min_axon = np.sqrt((meshes["axon"].vertices[:, 0] - row["x"]) ** 2 +
                                        (meshes["axon"].vertices[:, 1] - row["y"]) ** 2 +
                                        (meshes["axon"].vertices[:, 2] - row["z"]) ** 2).min()

                    d_min_dendrite = np.sqrt((meshes["dendrite"].vertices[:, 0] - row["x"]) ** 2 +
                                            (meshes["dendrite"].vertices[:, 1] - row["y"]) ** 2 +
                                            (meshes["dendrite"].vertices[:, 2] - row["z"]) ** 2).min()

                    if d_min_axon < d_min_dendrite:
                        df_swc.loc[i, "label"] = 2  # Axon
                    else:
                        df_swc.loc[i, "label"] = 3  # Dendrite

                # Load the mapped synapses
                pre_synapses = []
                post_synapses = []

                if (root_path / cell_name / folder_str / f"{cell_name}_presynapses{suffix_str}.csv").exists():

                    print('Presynapses found')

                    df_presynapses = pd.read_csv(root_path / cell_name / folder_str / f"{cell_name}_presynapses{suffix_str}.csv",
                                                comment='#', sep=' ', header=None, names=["postsynaptic_cell_id", "x", "y", "z", "radius"])

                    # If we do this in the original make sure to scale the synpapse locations (which usally are in nm)
                    if mapped_i == 0:
                        df_presynapses.loc[:, "x"] = df_presynapses["x"] * input_scale_x
                        df_presynapses.loc[:, "y"] = df_presynapses["y"] * input_scale_y
                        df_presynapses.loc[:, "z"] = df_presynapses["z"] * input_scale_z

                    # Find the points in the swc list with the minimal distance to the synapse location, and make them a synapse
                    for i, row in df_presynapses.iterrows():

                        dist = np.sqrt((df_swc["x"] - row["x"]) ** 2 +
                                    (df_swc["y"] - row["y"]) ** 2 +
                                    (df_swc["z"] - row["z"]) ** 2)

                        i_min = dist.argmin()

                        # Minimal distance needs to be small
                        if dist[i_min] < 10 * neurite_radius:
                            df_swc.loc[i_min, "label"] = 4  # Pre synapse
                            pre_synapses.append([int(row["postsynaptic_cell_id"]), int(df_swc.loc[i_min, "node_id"])])
                        else:
                            print("Postsynaptic cell not connected to swc. Presynapse too far away:", int(row["postsynaptic_cell_id"]), dist[i_min])
                            pre_synapses.append([int(row["postsynaptic_cell_id"]), -1])

                if (root_path / cell_name / folder_str / f"{cell_name}_postsynapses_mapped.csv").exists():

                    print('Postsynapses found')

                    df_postsynapses = pd.read_csv(root_path / cell_name / folder_str / f"{cell_name}_postsynapses{suffix_str}.csv",
                                                comment='#', sep=' ', header=None, names=["presynaptic_cell_id", "x", "y", "z", "radius"])

                    # If we do this in the original make sure to scale the synpapse locations (which usally are in nm)
                    if mapped_i == 0:
                        df_postsynapses.loc[:, "x"] = df_postsynapses["x"] * input_scale_x
                        df_postsynapses.loc[:, "y"] = df_postsynapses["y"] * input_scale_y
                        df_postsynapses.loc[:, "z"] = df_postsynapses["z"] * input_scale_z

                    for i, row in df_postsynapses.iterrows():

                        dist = np.sqrt((df_swc["x"] - row["x"]) ** 2 +
                                    (df_swc["y"] - row["y"]) ** 2 +
                                    (df_swc["z"] - row["z"]) ** 2)

                        i_min = dist.argmin()

                        # Minimal distance needs to be small
                        if dist[i_min] < 10 * neurite_radius:
                            df_swc.loc[i_min, "label"] = 5  # Postsynapse
                            post_synapses.append([int(row["presynaptic_cell_id"]), int(df_swc.loc[i_min, "node_id"])])
                        else:
                            print("Presynaptic cell not connected to swc. Postsynapse too far away.", int(row["presynaptic_cell_id"]), dist[i_min])
                            post_synapses.append([int(row["presynaptic_cell_id"]), -1])

            # In case we are just mapping an axon
            elif "soma" not in meshes_mapped.keys() and "dendrite" not in meshes_mapped.keys() and "axon" in meshes_mapped.keys():
                print("Making SWC, only based on saxon")
                # Skeletonize the axons s at 1.5 um precision
                skel = sk.skeletonize.by_teasar(meshes["axon"], inv_dist=skeletonize_inv_dist)

                # Remove some potential perpedicular branches that should not be there
                skel = sk.post.clean_up(skel)
                skel = skel.reindex()

                # Continue with the swc as a pandas dataframe
                df_swc = skel.swc

                # Make a standard axon/dendrite radius everywhere
                df_swc["radius"] = neurite_radius
                df_swc["label"] = 0

                # Repair gaps in the skeleton using navis
                x = navis.read_swc(df_swc)
                x = navis.heal_skeleton(x, method='ALL', max_dist=None, min_size=None, drop_disc=False,
                                        mask=None, inplace=False)

                # Save as a temporary swc
                f_swc_temp = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.swc')
                f_swc_temp.close()

                x.to_swc(f_swc_temp.name)

                # Read it as a dataframe for further processing
                df_swc = pd.read_csv(f_swc_temp.name, sep=" ",
                                     names=["node_id", 'label', 'x', 'y', 'z', 'radius', 'parent_id'],
                                     comment='#', header=None)

                # Delete temporary file
                os.remove(f_swc_temp.name)

                for i, row in df_swc.iterrows():
                    df_swc.loc[i, "label"] = 2  # Axon

                # Load the mapped synapses
                pre_synapses = []
                post_synapses = []

                if (root_path / cell_name / folder_str / f"{cell_name}_presynapses{suffix_str}.csv").exists():

                    print('Presynapses found')

                    df_presynapses = pd.read_csv(
                        root_path / cell_name / folder_str / f"{cell_name}_presynapses{suffix_str}.csv",
                        comment='#', sep=' ', header=None,
                        names=["postsynaptic_cell_id", "x", "y", "z", "radius"])

                    # If we do this in the original make sure to scale the synpapse locations (which usally are in nm)
                    if mapped_i == 0:
                        df_presynapses.loc[:, "x"] = df_presynapses["x"] * input_scale_x
                        df_presynapses.loc[:, "y"] = df_presynapses["y"] * input_scale_y
                        df_presynapses.loc[:, "z"] = df_presynapses["z"] * input_scale_z

                    # Find the points in the swc list with the minimal distance to the synapse location, and make them a synapse
                    for i, row in df_presynapses.iterrows():

                        dist = np.sqrt((df_swc["x"] - row["x"]) ** 2 +
                                       (df_swc["y"] - row["y"]) ** 2 +
                                       (df_swc["z"] - row["z"]) ** 2)

                        i_min = dist.argmin()

                        # Minimal distance needs to be small
                        if dist[i_min] < 10 * neurite_radius:
                            df_swc.loc[i_min, "label"] = 4  # Pre synapse
                            pre_synapses.append(
                                [int(row["postsynaptic_cell_id"]), int(df_swc.loc[i_min, "node_id"])])
                        else:
                            print("Postsynaptic cell not connected to swc. Presynapse too far away:",
                                  int(row["postsynaptic_cell_id"]), dist[i_min])
                            pre_synapses.append([int(row["postsynaptic_cell_id"]), -1])

                if (root_path / cell_name / folder_str / f"{cell_name}_postsynapses_mapped.csv").exists():

                    print('Postsynapses found')

                    df_postsynapses = pd.read_csv(
                        root_path / cell_name / folder_str / f"{cell_name}_postsynapses{suffix_str}.csv",
                        comment='#', sep=' ', header=None,
                        names=["presynaptic_cell_id", "x", "y", "z", "radius"])

                    # If we do this in the original make sure to scale the synpapse locations (which usally are in nm)
                    if mapped_i == 0:
                        df_postsynapses.loc[:, "x"] = df_postsynapses["x"] * input_scale_x
                        df_postsynapses.loc[:, "y"] = df_postsynapses["y"] * input_scale_y
                        df_postsynapses.loc[:, "z"] = df_postsynapses["z"] * input_scale_z

                    for i, row in df_postsynapses.iterrows():

                        dist = np.sqrt((df_swc["x"] - row["x"]) ** 2 +
                                       (df_swc["y"] - row["y"]) ** 2 +
                                       (df_swc["z"] - row["z"]) ** 2)

                        i_min = dist.argmin()

                        # Minimal distance needs to be small
                        if dist[i_min] < 10 * neurite_radius:
                            df_swc.loc[i_min, "label"] = 5  # Postsynapse
                            post_synapses.append(
                                [int(row["presynaptic_cell_id"]), int(df_swc.loc[i_min, "node_id"])])
                        else:
                            print("Presynaptic cell not connected to swc. Postsynapse too far away.",
                                  int(row["presynaptic_cell_id"]), dist[i_min])
                            post_synapses.append([int(row["presynaptic_cell_id"]), -1])

            # Reorder columns for proper storage
            df_swc = df_swc.reindex(columns=['node_id', 'label', "x", "y", "z", 'radius', 'parent_id'])

            metadata["presynapses"] = pre_synapses
            metadata["postsynapses"] = post_synapses

            # Save the swc
            header = (f"# SWC format file based on specifications at http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html\n"
                    f"# Generated by 'map_and_skeletonize_cell' of the ANTs registration helper library developed by the Bahl lab Konstanz.\n"
                    f"# Metadata: {str(metadata)}\n"
                    f"# Labels: 0 = undefined; 1 = soma; 2 = axon; 3 = dendrite; 4 = Presynapse; 5 = Postsynapse\n")

            with open(root_path / cell_name / folder_str / f"{cell_name}{suffix_str}.swc", 'w') as fp:
                fp.write(header)
                df_swc.to_csv(fp, index=False, sep=' ', header=None)

    def convert_synapse_file(self, root_path, cell_name,
                             shift_x,
                             shift_y,
                             shift_z,
                             scale_x,
                             scale_y,
                             scale_z,
                             radius_set):

        if not (root_path / cell_name / f"{cell_name}_synapses.txt").exists():
            print(cell_name, "No synpase file found.")
            return

        fp = open(root_path / cell_name / f"{cell_name}_synapses.txt", 'r')

        split_data = fp.read().split(",postsynaptic")

        presynaptic_data_str = split_data[0]
        postsynaptic_data_str = split_data[1]

        presynaptic_data_str = presynaptic_data_str.replace("'", "")
        postsynaptic_data_str = postsynaptic_data_str.replace("'", "")

        for i, data_str in enumerate([presynaptic_data_str, postsynaptic_data_str]):

            # Extracting presynaptic data
            start_index = data_str.find("[")
            end_index = data_str.find("]")
            synaptic_data = data_str[start_index + 1:end_index]

            # Splitting the presynaptic data into individual entries
            synaptic_list = synaptic_data.split(", ")

            # Creating a list of dictionaries to represent the table
            table_data = []
            for entry in synaptic_list:
                if entry != "[]" and entry != "":
                    values = entry.split(",")
                    table_data.append({
                        'partner_cell_id': str(values[0]),
                        'x': int(values[1].split(".")[0]),
                        'y': int(values[2].split(".")[0]),
                        'z': int(values[3].split(".")[0]),
                        'Size': radius_set
                    })

            # Converting the list of dictionaries into a pandas DataFrame
            df = pd.DataFrame(table_data)

            if len(df) > 0:
                # Apply the conversion from highres to lowres
                df.loc[:, "x"] = shift_x + df["x"] * scale_x
                df.loc[:, "y"] = shift_y + df["y"] * scale_y
                df.loc[:, "z"] = shift_z + df["z"] * scale_z

            if i == 0:
                df.to_csv(root_path / cell_name / f"{cell_name}_presynapses.csv", index=False, sep=' ', header=None, float_format='%.8f')
            else:
                df.to_csv(root_path / cell_name / f"{cell_name}_postsynapses.csv", index=False, sep=' ', header=None, float_format='%.8f')
