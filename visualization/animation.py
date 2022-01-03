import glob
from plotly.graph_objs import Figure
import plotly.graph_objects as go

from meshlib import Mesh
from render.plot import BrowserVisualizer
from typing import Sequence

def animate(vert_path, face_path):
    targetPattern = "/*.txt"
    path_list = glob.glob(vert_path + targetPattern)

    mesh_list = []
    for i in range(len(path_list)):
        mesh_list.append(Mesh.load_txt(vert_file = path_list[i], face_file = face_path))
    fig = make_animation(mesh_list)
    fig.show(renderer="browser")

def make_animation(mesh_list):
    mesh_kwargs = dict(
        color='#ccc',
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
    fig = Figure(
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
    # cfg = ConfigFile.load(ConfigFile.Paths.highpoly.horse_camel)
    path = "C:/Users/Enzo.Magal/Documents/Enzo2021/AlphSistant/visualization/3Dmodels/test/source_obj"
    # cfg = ConfigFile.load(ConfigFile.Paths.highpoly.cat_lion)
    animate(path)