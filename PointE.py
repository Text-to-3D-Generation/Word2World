import numpy as np
import torch
from tqdm.auto import tqdm

from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.plotting import plot_point_cloud

def rotation_matrix_to_quaternion(R):
    """
    Convert a 3x3 rotation matrix R into a quaternion [w, x, y, z].
    """
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    
    trace = m00 + m11 + m22
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    elif (m00 > m11) and (m00 > m22):
        s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    return np.array([w, x, y, z])

class PointEModel:
    def __init__(self, n_base=2048, n_upsampler=2048, n_iters=130):
        """
        Initializes the Point-E model.
        
        Args:
            n_base: The number of points for the base model.
            n_upsampler: The number of points for the upsampler model.
            n_iters: The number of iterations for sample production (max=130).
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_base = n_base
        self.n_upsampler = n_upsampler
        self.n_iters = min(n_iters, 130)

        # Load the base model
        print("Creating base model...")
        base_name = 'base40M-textvec'
        self.base_model = model_from_config(MODEL_CONFIGS[base_name], self.device)
        self.base_model.eval()
        self.base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])
        self.base_model.load_state_dict(load_checkpoint(base_name, self.device))

        # Load the upsampler model        
        print("Creating upsampler model...")
        self.upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], self.device)
        self.upsampler_model.eval()
        self.upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])
        self.upsampler_model.load_state_dict(load_checkpoint('upsample', self.device))

        # Create the sampler
        print("Creating sampler model...")
        self.sampler = PointCloudSampler(
            device=self.device,
            models=[self.base_model, self.upsampler_model],
            diffusions=[self.base_diffusion, self.upsampler_diffusion],
            num_points=[self.n_base, self.n_upsampler],
            aux_channels=['R', 'G', 'B'],
            guidance_scale=[3.0, 0.0],
            model_kwargs_key_filter=('texts', ''), 
        )

    def random_rotation_matrix(self):
        """Generate a random 3D rotation matrix using QR decomposition."""
        H = np.random.randn(3, 3)  # Random matrix
        Q, R = np.linalg.qr(H)     # QR decomposition gives an orthonormal matrix
        return Q

    def to_gaussians(self, pc, std_dev):
        """
        Generate Gaussian samples for each point in the given array.
        Instead of computing a full covariance matrix, we store only the svec and qvec.
        """
        all_samples = []
        # For an isotropic Gaussian, the svec is just [std_dev, std_dev, std_dev]
        for i, mean in enumerate(pc.coords):
            R = self.random_rotation_matrix()
            qvec = rotation_matrix_to_quaternion(R)
            svec = np.array([std_dev, std_dev, std_dev])
            random_opacity = np.random.uniform(0.0, 1.0)
            color = np.array([
                pc.channels['R'][i],
                pc.channels['G'][i],
                pc.channels['B'][i]
            ])
            # Store (mean, qvec, svec, color, alpha) instead of (mean, covariance, color, alpha)
            samples = (mean, qvec, svec, color, random_opacity)
            all_samples.append(samples)
        return all_samples
    
    def generate_point_cloud(self, prompt):
        """
        Generates a point cloud from a text prompt.
        
        Args:
            prompt: The text prompt to condition on.
        
        Returns:
            A point cloud as a numpy array.
        """
        samples = None
        max_no_of_productions = 0
        for x in tqdm(self.sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(texts=[prompt]))):
            samples = x
            if max_no_of_productions >= self.n_iters:
                break
            max_no_of_productions += 1
        
        pc = self.sampler.output_to_point_clouds(samples)[0]
        return pc
    
    def generate_gaussians(self, prompt):
        """
        Generates Gaussian samples from a text prompt.
        
        Args:
            prompt: The text prompt to condition on.
        
        Returns:
            A list of Gaussian samples.
        """
        pc = self.generate_point_cloud(prompt)
        return self.to_gaussians(pc, std_dev=0.5)
    
    def plot_point_cloud(self, pc):
        """
        Plots a point cloud.
        
        Args:
            pc: The point cloud to plot.
        """
        plot_point_cloud(pc, grid_size=3, fixed_bounds=((-0.75, -0.75, -0.75),
                                                         (0.75, 0.75, 0.75)))
    
    def save_checkpoint(self, prompt, filename, std_dev=0.5):
        """
        Generates Gaussian samples from a prompt, converts them to the expected format,
        and saves a checkpoint containing "mean", "qvec", "svec", "color", and "alpha".
        
        Args:
            prompt: Text prompt used to generate the Gaussians.
            filename: Path where the checkpoint will be saved.
            std_dev: The standard deviation used in generating the Gaussians.
        """
        gaussians = self.generate_gaussians(prompt)
        
        means = []
        qvecs = []
        svecs = []
        colors = []
        alphas = []
        
        for (mean, qvec, svec, color, alpha) in gaussians:
            means.append(torch.tensor(mean, dtype=torch.float32))
            qvecs.append(torch.tensor(qvec, dtype=torch.float32))
            svecs.append(torch.tensor(svec, dtype=torch.float32))
            colors.append(torch.tensor(color, dtype=torch.float32))
            alphas.append(torch.tensor(alpha, dtype=torch.float32))
        
        checkpoint = {
            "cfg": {"prompt": {"prompt": prompt}},
            "mean": torch.stack(means),
            "qvec": torch.stack(qvecs),
            "svec": torch.stack(svecs),
            "color": torch.stack(colors),
            "alpha": torch.stack(alphas),
        }
        
        torch.save(checkpoint, filename)
        print(f"Checkpoint saved to {filename}")
