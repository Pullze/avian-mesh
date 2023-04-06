from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    PerspectiveCameras,
    DirectionalLights, 
    AmbientLights,
    RasterizationSettings, 
    FoVPerspectiveCameras,
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesVertex,
    look_at_view_transform
)
import torch
from matplotlib import pyplot as plt
from utils.geometry import perspective_projection

class RendererP3D:
    def __init__(self, focal_length=2167, faces=None, center=None, img_size=256, background=None) -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.focal_length = focal_length
        self.faces = torch.IntTensor(faces).to(self.device)

    
        # E = torch.FloatTensor([[1, 0, 0, 0],
        #                         [0, -1, 0, 0],
        #                         [0, 0, -1, 0],
        #                         [0, 0, 0, 1]])
        
        self.K = torch.FloatTensor([
            [2167,   0,   128,   0],
            [0,   2167,   128,   0],
            [0,    0,    1,   1],
            [0, 0, 1, 0]])

        self.R = torch.FloatTensor([[1, 0, 0],
                            [0, -1, 0],
                            [0, 0, -1]])
        self.T = torch.zeros((1, 3))
        
        # self.Proj = E @ self.K
        # print(self.Proj)
        
        # self.cameras = FoVPerspectiveCameras(
        # R=torch.FloatTensor([[1, 0, 0],
        #                     [0, -1, 0],
        #                     [0, 0, -1]]).unsqueeze(0), T=torch.zeros((1, 3)), 
        #                     zfar=1000, fov=7)
        
        self.cameras = PerspectiveCameras(K=self.K.unsqueeze(0),
                                          R=self.R.unsqueeze(0), 
                                          T=self.T,
                                          in_ndc=False, 
                                          image_size=[(256,256)], device=self.device)

        # print(cameras)
        # R, T = look_at_view_transform(30, 0, 180) 
        # self.cameras = FoVPerspectiveCameras(R=R, T=T)
        
        # print(self.cameras.get_full_projection_transform().get_matrix())
        # print(self.cameras.get_projection_transform().get_matrix())
        # print(self.cameras.get_world_to_view_transform().get_matrix())
        # print(self.cameras.get_full_projection_transform().transform_points(torch.tensor([[-7.3891, -4.0384, -1.5935],
        #  [-4.5288, -3.4811, -2.4584],
        #  [-4.6276, -4.8838, -0.6452],
        #  [-5.1463, -2.5929, -0.2575],
        #  [-1.2442, -3.3303, -1.0593],
        #  [ 8.0212,  2.9161, -1.6047],
        #  [ 8.2726,  2.0692, -1.4950],
        #  [-1.2360,  5.3669,  1.8540],
        #  [ 0.4082,  3.5080,  4.4820],
        #  [ 8.1178,  7.7434,  1.0640],
        #  [-3.0649, -1.6582, -2.3060],
        #  [-2.0782, -4.1007,  1.1512]])))
        
        lights = DirectionalLights(device=self.device)
        
        raster_settings = RasterizationSettings(
            image_size=img_size, 
            blur_radius=0.0, 
            faces_per_pixel=1, 
        )
        
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras, 
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                cameras=self.cameras,
                lights=lights,
                device=self.device
            )
        )  
    def __call__(self, vertices):
        
        # print(self.faces.shape, vertices.shape)
        
        color = torch.ones(1, vertices.shape[0], 3) * 0.9
        color = color.to(self.device)
        
        # print(color.shape)
        
        vertices[:, 2] *= -1
        vertices[:, 0] *= -1
        
        vertices = vertices.to(self.device)

        textures = TexturesVertex(verts_features=color)
        mesh = Meshes(verts=[vertices], faces=[self.faces], textures=textures)
        
        # fig = plot_scene({
        #     "subplot1": {
        #         "cow_mesh": Meshes(verts=[vertices], faces=[self.faces]),
        #         # "camera": self.cameras
        #     },
        # }) #, viewpoint_cameras=self.cameras)
        # fig.show()
        
        output = self.renderer(mesh, zfar=1000)
        img = output[0, ..., :3].detach()
        mask = output[0, ..., 3].detach()
        plt.imsave("./233.jpg", img.cpu().numpy())
        mask[mask > 0] = 1
    
        return img, mask

    # def project(self, vertices):
