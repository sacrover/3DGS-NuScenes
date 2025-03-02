import cv2

from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib import pyplot as plt

import numpy as np

from nuscenes.utils.data_classes import LidarPointCloud
import open3d as o3d

from scipy.interpolate import griddata

import plotly.graph_objs as go
from plotly.subplots import make_subplots


def plot_fig_legend(fig, color_cycle):
    class_labels = [
        'void', 'barrier', 'bicycle', 'bus', 'car', 'construction vehicle', 'motorcycle', 'pedestrian',
        'traffic cone', 'trailer', 'truck', 'drivable surface', 'other flat',
        'sidewalk', 'terrain', 'manmade', 'vegetation']

    legend_elements = []

    for idx, label in enumerate(class_labels):
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label=label,
                                      markerfacecolor=color_cycle[idx], markersize=8))

    fig.legend(handles=legend_elements, title='Class', loc='center', bbox_to_anchor=(0.5, 0), fontsize='11', ncol=2)


def plot_lidar_projection(ax, img, segment=False, color_cycle=None, filtered=None, class_filtered=None, point_size=5):
    ax.imshow(img)

    if segment:
        class_colors = {}
        for idx in range(filtered.shape[1]):
            class_colors[idx] = color_cycle[int(class_filtered[idx])]

        colors = [color_cycle[int(class_id)] for class_id in class_filtered]

        ax.scatter(filtered[0, :], filtered[1, :], s=point_size, c=colors)

    else:
        ax.scatter(filtered[0, :], filtered[1, :], s=point_size)


def visualise_3d_pcl(pcl_path, annotated_labels):
    pc = LidarPointCloud.from_file(pcl_path)  # Random 3D points

    # Define a color map for the label classes
    class_colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'cyan', 'magenta', 'teal', 'gray', 'pink',
                    'lightblue', 'lightgreen', 'lightgray', 'darkblue', 'brown', 'darkgreen', 'red']
    cmap = ListedColormap(class_colors)

    # Assign colors to each point based on label class
    colors = cmap(annotated_labels)

    # Visualize the colored point cloud in 3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc.points[:3].T)
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])  # Normalize colors to range [0, 1]

    o3d.visualization.draw_geometries([pcd])

def dense_depth_map(depth_map_sparse, n, m, grid):
    """
    https://  github.com/balcilar/DenseDepthMap
    """
    ng = 2 * grid + 1
    epsilon = 1e-6  # Small value to prevent division by zero

    # Find the non-zero elements in the sparse depth map
    non_zero_indices = np.nonzero(depth_map_sparse)
    print("non_zero_indices shape", non_zero_indices[0])
    Pts = np.vstack((non_zero_indices[1], non_zero_indices[0], depth_map_sparse[non_zero_indices])).T
    print("Pts shape",  Pts.shape)
    print("Pts", Pts)
    # Convert sparse points to indices
    linear_index = np.ravel_multi_index(
        (Pts[:, 1].astype(int), Pts[:, 0].astype(int)), (n, m)
    )
    print("linear_index shape", linear_index.shape)
    # Initialize matrices
    mX = np.full((n, m), np.inf)
    mY = np.full((n, m), np.inf)
    mD = np.zeros((n, m))

    # Populate matrices with point data
    mX.flat[linear_index] = Pts[:, 0] - np.floor(Pts[:, 0])
    mY.flat[linear_index] = Pts[:, 1] - np.floor(Pts[:, 1])
    mD.flat[linear_index] = Pts[:, 2]

    print("mx flat shape", mX.flat)

    # Prepare for grid-based interpolation
    KmX, KmY, KmD = [], [], []
    for i in range(ng):
        for j in range(ng):
            KmX.append(mX[i : n - ng + i + 1, j : m - ng + j + 1] - grid - 1 + i)
            KmY.append(mY[i : n - ng + i + 1, j : m - ng + j + 1] - grid - 1 + j)
            KmD.append(mD[i : n - ng + i + 1, j : m - ng + j + 1])

    # Perform weighted interpolation
    S = np.zeros_like(KmD[0])
    Y = np.zeros_like(KmD[0])
    for i in range(ng):
        for j in range(ng):
            s = 1.0 / (np.sqrt(KmX[i * ng + j] ** 2 + KmY[i * ng + j] ** 2) + epsilon)
            Y += s * KmD[i * ng + j]
            S += s

    S[S == 0] = 1  # Avoid division by zero
    out = np.zeros((n, m))
    out[grid : n - grid, grid : m - grid] = Y / S

    # Replace NaN and inf values with 0
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    return out


def plot_depth_map(width, height, filtered_points, filtered_depths, plot=False, ax=None, fig=None, save_path=None):
    # Depth map (sparse)
    # sparse = np.hstack((filtered_points, filtered_depths.reshape(-1, 1)))
    depth_map_sparse = np.zeros((height, width))
    depth_map_sparse[filtered_points[:, 1].astype(np.uint16), filtered_points[:, 0].astype(np.uint16)] = filtered_depths


    # Interpolate depth map (dense)

    # depth_map_dense = dense_depth_map(depth_map_sparse, height, width, grid=7)
    depth_map_dense = depth_map_sparse

    # Normalize interpolated depth map
    min_depth, max_depth = 1.0, 100.0
    depth_map_dense = (depth_map_dense - min_depth) / (max_depth - min_depth)
    depth_map_dense = np.clip(depth_map_dense, 0, 1)

    # Mask the result to the Region of Interest (ROI)
    mask = np.ones((height, width))
    # ind = (depth_map_dense != 0).argmax(axis=0)
    # for i in range(width):
    #     mask[ind[i]:, i] = 1

    # kernel = np.ones((3, 3), np.uint8)

    kernel = np.array([[0.25, 0.25, 0.25], [0.25, 1, 0.25], [0.25, 0.25, 0.25]])

    dilate_dense_map = cv2.dilate(depth_map_dense, kernel, iterations=4)  # Dilation

    # Mask the depth map
    masked_depth_map = (mask * 255 * dilate_dense_map).astype(np.uint8)
    # masked_depth_map = (mask * 255 * depth_map_dense).astype(np.uint8)

    # Plot the depth map
    if plot:
        im = ax.imshow(masked_depth_map, cmap="jet", vmin=0, vmax=255)
        cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.012, ax.get_position().height])
        cbar = plt.colorbar(im, cax=cax)  # Similar to fig.colorbar(im, cax = cax)
        cbar.ax.set_ylabel(r"Normalized Depth", labelpad=20, rotation=270)
        ax.set_axis_off()


    if save_path is not None:
        plt.imsave(save_path, masked_depth_map, cmap='gray')


def plot_3d_scatter(data, labels):
    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])

    for i, array in enumerate(data):
        x = array[0]
        y = array[1]
        z = array[2]
        fig.add_trace(go.Scatter3d(x=[x], y=[y], z=[z], mode='markers', name=labels[i]))

    fig.update_layout(scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
        aspectmode='cube',
        camera_eye=dict(x=0, y=0, z=2),
    ),
        margin=dict(l=0, r=0, b=0, t=0),
    )

    fig.show()


def plot_3D_axes(fig, translation, rotation):
    # Plot camera position

    # Plot camera orientation axes
    origin = translation
    for i in range(3):
        axis_direction = rotation[:, i]
        axis_direction_normalized = axis_direction / np.linalg.norm(axis_direction)  # Normalize axis direction vector
        scaled_axis_direction = 0.15 * axis_direction_normalized  # Scale down the direction vector
        x_values = [origin[0], origin[0] + scaled_axis_direction[0]]
        y_values = [origin[1], origin[1] + scaled_axis_direction[1]]
        z_values = [origin[2], origin[2] + scaled_axis_direction[2]]
        axis_color = 'red' if i == 0 else 'green' if i == 1 else 'blue'  # Assign color based on axis
        fig.add_trace(
            go.Scatter3d(x=x_values, y=y_values, z=z_values, mode='lines', line=dict(color=axis_color, width=3),
                         name=f'Axis {i + 1}', showlegend=False))

        # Plot quiver
        fig.add_trace(go.Scatter3d(x=[origin[0] + scaled_axis_direction[0]], y=[origin[1] + scaled_axis_direction[1]],
                                   z=[origin[2] + scaled_axis_direction[2]],
                                   mode='markers', marker=dict(size=1), showlegend=False))


def plot_frustum(fig, translation, rotation, color='rgba(255, 0, 0, 0.2)'):
    # Define frustum vertices
    vertices = np.array([[1, 0.6, 1.5], [-1, 0.6, 1.5], [-1, -0.6, 1.5], [1, -0.6, 1.5], [0, 0, 0]]) * 0.1

    # Transform vertices to camera space
    transformed_vertices = (rotation @ vertices.T).T + translation

    # Define pyramid faces
    faces = [[0, 1, 2, 3], [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]

    # Plot pyramid faces
    for face in faces:
        x_values = transformed_vertices[face, 0]
        y_values = transformed_vertices[face, 1]
        z_values = transformed_vertices[face, 2]
        fig.add_trace(go.Scatter3d(x=x_values, y=y_values, z=z_values, mode='lines', line=dict(color='black', width=2),
                                   showlegend=False))
        fig.add_trace(go.Mesh3d(x=x_values, y=y_values, z=z_values, color=color, showlegend=False))


def plot_camera_pose(extrinsic_matrices, sensors, only_plot_obj=False, fig=None, color='rgba(255, 0, 0, 0.2)'):
    """
    Visualizes the extrinsics of multiple cameras in 3D space as pyramids (frustum).

    Parameters:
    - extrinsic_matrices (numpy.ndarray): Array of camera extrinsics matrices (Nx4x4).
    - sensors (list): List of sensor names (used for Legend).
    """
    fig = fig if fig is not None else make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])

    legend_size = len([d['legendgroup'] for d in fig.data if d['legendgroup'] == "main"])

    for idx, camera_extrinsics in enumerate(extrinsic_matrices):

        legend_id = legend_size + idx
        #get transform from sensor frame to global frame
        T_gs = camera_extrinsics

        # Extract translation and rotation from camera extrinsics matrix
        translation = T_gs[:3, 3]
        rotation_matrix = T_gs[:3, :3]

        fig.add_trace(go.Scatter3d(x=[translation[0]], y=[translation[1]], z=[translation[2]], mode='markers',
                                   name=f'{legend_id}:{sensors[idx]}',
                                   marker=dict(size=4, color=color), showlegend=True, legendgroup="main",
                                   legendgrouptitle_text="Sensors"))
        if sensors[idx] not in ['LIDAR_TOP']:  # TODO Add other sensors to omit frustum plot
            plot_frustum(fig, translation, rotation_matrix, color)
        plot_3D_axes(fig, translation, rotation_matrix)

        annotation_pos = T_gs @ np.array([0.0, 0.0, 0.15, 1.0]).T
        annotation_x = annotation_pos[0]
        annotation_y = annotation_pos[1]
        annotation_z = annotation_pos[2]
        fig.add_trace(go.Scatter3d(x=[annotation_x], y=[annotation_y], z=[annotation_z], mode='text', text=[legend_id],
                                   textposition='middle center', showlegend=False,
                                   textfont=dict(size=20, color='black', family='Arial')))

    fig.update_layout(scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        # zaxis_title='Z',
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        aspectmode='data',
        aspectratio=dict(x=1, y=1, z=0.5)),
        margin=dict(l=0, r=0, b=0, t=0), )
    if not only_plot_obj:
        fig.show()

    return fig
