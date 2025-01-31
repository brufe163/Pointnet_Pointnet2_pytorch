import os
import argparse
import numpy as np
import json
from ouster.sdk import client, pcap
from itertools import islice

def pcap_to_npy(filename, num_frames, output_folder):
    """
    Write point cloud data from a pcap to separate .npy files (coordinates, intensity, reflectivity).
    Each frame will be saved in its own folder.
    """
    pcap_file = filename + '.pcap'
    json_file = filename + '.json'
    # Load Ouster sensor config from JSON file
    with open(json_file, 'r') as f:
        metadata = client.SensorInfo(f.read())  # Load sensor metadata from JSON file
    
    # Create a PCAP reader
    source = pcap.Pcap(pcap_file, metadata)  # Read the pcap file
    
    # Precompute xyzlut to save computation in a loop
    xyzlut = client.XYZLut(metadata)
    
    # Create an iterator of LidarScans from pcap and bound it if num_frames is specified
    scans = iter(client.Scans(source))
    if num_frames:
        scans = islice(scans, num_frames)
    
    for idx, scan in enumerate(scans):
        # Create a folder for each frame
        frame_folder = os.path.join(output_folder, f'amtc_{idx}')
        os.makedirs(frame_folder, exist_ok=True)
        
        # Extract point cloud data
        xyz_1 = xyzlut(scan.field(client.ChanField.RANGE)).reshape(-1,3)
        xyz_2 = xyzlut(scan.field(client.ChanField.RANGE2)).reshape(-1,3)
        intensity_1 = scan.field(client.ChanField.SIGNAL).flatten()
        intensity_2 = scan.field(client.ChanField.SIGNAL2).flatten()
        reflectivity_1 = scan.field(client.ChanField.REFLECTIVITY).flatten()
        reflectivity_2 = scan.field(client.ChanField.REFLECTIVITY2).flatten()

        # Union of the two point clouds and intensities
        coord_12 = np.vstack((xyz_1, xyz_2))
        intensity_12 = np.concatenate((intensity_1, intensity_2))
        reflectivity_12 = np.concatenate((reflectivity_1, reflectivity_2))
        
        # Ensure intensity and reflectivity match xyz shape
        # num_points = xyz_1.shape[0]
        # if intensity_1.shape[0] != num_points:
        #     print(f"Warning: Intensity shape mismatch ({intensity.shape[0]} != {num_points})")
        #     intensity = np.interp(np.arange(num_points), np.linspace(0, num_points, intensity.shape[0]), intensity)
        # if reflectivity.shape[0] != num_points:
        #     print(f"Warning: Reflectivity shape mismatch ({reflectivity.shape[0]} != {num_points})")
        #     reflectivity = np.interp(np.arange(num_points), np.linspace(0, num_points, reflectivity.shape[0]), reflectivity)
        
        # Save each array to its corresponding .npy file
        np.save(os.path.join(frame_folder, 'coord_1.npy'), xyz_1)
        np.save(os.path.join(frame_folder, 'coord_2.npy'), xyz_2)
        np.save(os.path.join(frame_folder, 'coord_12.npy'), coord_12)
        
        np.save(os.path.join(frame_folder, 'intensity_1.npy'), intensity_1)
        np.save(os.path.join(frame_folder, 'intensity_2.npy'), intensity_2)
        np.save(os.path.join(frame_folder, 'intensity_12.npy'), intensity_12)
        
        np.save(os.path.join(frame_folder, 'reflectivity_1.npy'), reflectivity_1)
        np.save(os.path.join(frame_folder, 'reflectivity_2.npy'), reflectivity_2)
        np.save(os.path.join(frame_folder, 'reflectivity_12.npy'), reflectivity_12)
        
        print(f"Saved frame {idx + 1} to {frame_folder}")

    print(f"Finished saving {num_frames} frames to {output_folder}")

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert Ouster PCAP and JSON to HDF5.')
    parser.add_argument('--filename', type=str, help='Path to the file + filename (without extension!!!!).')
    #parser.add_argument('--json_file', type=str, help='Path to the input JSON file.')
    parser.add_argument('--num_frames', type=int, default=0, help='Number of frames to process (default is all frames).')
    parser.add_argument('--output_folder', type=str, default='/home/nicolas/repos/custom_pointnet2_pytorch/data/ouster_data/Area_1', help='Path to save the output HDF5 file (default is output.h5).')
    
    args = parser.parse_args()
    
    pcap_to_npy(args.filename, args.num_frames, args.output_folder)
