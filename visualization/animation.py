import glob
import sys
from plotly.graph_objs import Figure
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.append("C:/Users/Enzo.Magal/Documents/Enzo2021/AlphSistant/visualization")
from visualization.config import ConfigFile
from visualization.transformation import Transformation
from visualization.correspondence import get_correspondence
from meshlib import Mesh
from render.plot import BrowserVisualizer
from typing import Sequence

def animate(vert_path, face_path, cfg: ConfigFile, identity=False):
    targetPattern = "/*.txt"
    path_list = glob.glob(vert_path + targetPattern)

    mesh_list = []
    for i in range(len(path_list)):
        mesh_list.append(Mesh.load_txt(vert_file = path_list[i], face_file = face_path))

    corr_markers = cfg.markers  # List of vertex-tuples (source, target)
    if identity:
        corr_markers = np.ascontiguousarray(np.array((corr_markers[:, 0], corr_markers[:, 0]), dtype=np.int).T)

    original_source = Mesh.load(cfg.source.reference)
    original_target = Mesh.load(cfg.target.reference)
    if identity:
        original_target = Mesh.load(cfg.source.reference)

    mapping = get_correspondence(original_source, original_target, corr_markers)
    transf = Transformation(original_source, original_target, mapping, smoothness=1)

    fig = make_animation(transf, list(cfg.source.load_poses()), mesh_list)
    fig.show(renderer="browser")

def make_animation(transf: Transformation, poses: Sequence[Mesh], mesh_list):
    assert poses
    results = [transf(pose) for pose in poses]

    mesh_kwargs = dict(
        color='#666',
        opacity=1.0,
        flatshading=True,
        lighting=dict(
            ambient=0.1,
            diffuse=1.0,
            facenormalsepsilon=0.0000000000001,
            roughness=0.3,
            specular=0.7,
            fresnel=0.001
        ),
        lightposition=dict(
            x=-10000,
            y=10000,
            z=5000
        )
    )

    fig = make_subplots(
    rows=1, cols=2, subplot_titles=('Prédiction', 'Retargeting'),
    specs=[[{'type': 'surface'}, {'type': 'surface'}]])

    fig1 = Figure(
        data=[BrowserVisualizer.make_mesh(mesh_list[0], **mesh_kwargs)],
        layout=dict(
            updatemenus=[
                dict(type="buttons",
                     buttons=[
                         dict(
                             label="Play",
                             method="animate",
                             args=[None, {
                                 "mode": "afterall",
                                 "frame": {"duration": 40, "redraw": True},
                                 "fromcurrent": False,
                                 "transition": {"duration": 40, "easing": "linear", "ordering": "traces first"}
                             }]
                         )
                     ])
            ],
        ),
        frames=[go.Frame(data=[BrowserVisualizer.make_mesh(mesh_list[i], **mesh_kwargs)]) for i in range(len(mesh_list))]
    )

    fig2 = Figure(
        data=[BrowserVisualizer.make_mesh(results[0].transpose((0, 2, 1)), **mesh_kwargs)],
        layout=dict(
            updatemenus=[
                dict(type="buttons",
                     buttons=[
                         dict(
                             label="Play",
                             method="animate",
                             args=[None, {
                                 "mode": "afterall",
                                 "frame": {"duration": 40, "redraw": True},
                                 "fromcurrent": False,
                                 "transition": {"duration": 40, "easing": "linear", "ordering": "traces first"}
                             }]
                         )
                     ])
            ],
        ),
        frames=[go.Frame(data=[BrowserVisualizer.make_mesh(p.transpose((0, 2, 1)), **mesh_kwargs)]) for p in results]
    )

    fig.append_trace(fig1['data'][0], 1, 1)
    fig.append_trace(fig2['data'][0], 1, 2)

    frames=[go.Frame(data=[BrowserVisualizer.make_mesh(mesh_list[i], **mesh_kwargs),
                            BrowserVisualizer.make_mesh(results[i].transpose((0, 2, 1)), **mesh_kwargs)])
                             for i in range(min(len(results), len(mesh_list)))]
    fig.frames=frames
    button = dict(
                 label='Play',
                 method='animate',
                 args=[None, {
                                 "mode": "afterall",
                                 "frame": {"duration": 40, "redraw": True},
                                 "fromcurrent": False,
                                 "transition": {"duration": 40, "easing": "linear", "ordering": "traces first"}
                             }])
    fig.update_layout(updatemenus=[dict(type='buttons', buttons=[button])])

    camera = dict(
        up=dict(x=0, y=1, z=0)
    )
    scene = dict(
        aspectmode='data',
        xaxis_title='X',
        yaxis_title='Z',
        zaxis_title='Y',
        camera=camera,
        dragmode='turntable'
    )
    fig.update_layout(
        scene=scene,
        scene2=scene,
        yaxis=dict(scaleanchor="x", scaleratio=1),
        yaxis2=dict(scaleanchor="x", scaleratio=1),
        margin=dict(l=0, r=0),
        # scene_camera=camera
    )
    return fig  

if __name__ == "__main__":
    vertice_file_path = "C:/Users/Enzo.Magal/Documents/Enzo2021/AlphSistant/prediction"
    face_file_path = "C:/Users/Enzo.Magal/Documents/Enzo2021/AlphSistant/alphsistant_face_tris.txt"

    cfg = ConfigFile.load("C:/Users/Enzo.Magal/Documents/Enzo2021/AlphSistant/fixed_map.yml")
    #cfg = ConfigFile.load("C:/Users/Enzo.Magal/Documents/Enzo2021/alphsistant_code/deformation_external/models/lowpoly/markers-cat-voxel.yml")
    animate(vert_path, face_path, cfg)