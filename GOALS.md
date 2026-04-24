The goal is to:
• Generate multi-view renderings of a 3D object using PyTorch3D
• Extract silhouettes and camera parameters
• Implement space carving under the pinhole camera model
• Reconstruct the visual hull of the object
• Convert the visual hull into a mesh using Open3D

2 Background
2.1 Visual Hull and Space Carving
The visual hull is the maximal 3D volume consistent with all object silhouettes across
views. It is computed by:
1. Starting with a 3D volume (voxel grid)
2. Projecting each voxel into all camera views
3. Removing voxels that fall outside any silhouette
This process is called space carving.
2.2 Camera Model
All projections must follow the pinhole camera model, where:
• 3D points are projected using intrinsic and extrinsic parameters
• Consistency between rendering and projection is critical
3 Provided Resources
You must adapt and reuse the following notebooks:
• Rendering and camera generation:
https://github.com/ribeiro-computer-vision/pytorch3d_rendering/blob/main/demo_
pytorch3D_rendering.ipynb
• Camera model and projection math:
https://github.com/ribeiro-computer-vision/pinhole_camera_model/blob/main/
the_pinhole_camera_model.ipynb
• Surface reconstruction reference:
https://www.open3d.org/docs/latest/tutorial/Advanced/surface_reconstruction.
html

4 Tasks
4.1 Part 1 — Multi-View Rendering (PyTorch3D)
1. Load a 3D mesh (e.g., OBJ file)
2. Generate multiple views (recommended: 20–50 views)
3. For each view, extract:
• RGB image
• Silhouette (binary mask)
• Depth map (optional)
• Camera parameters:
– Intrinsics: fx, fy, cx, cy
– Extrinsics: R, T
Output: Dataset of images, silhouettes, and camera parameters.
4.2 Part 2 — Projection Using the Pinhole Model
1. Implement projection:
x = K[R | T]X
2. Convert 3D voxel coordinates into pixel coordinates
3. Handle:
• Homogeneous normalization
• Visibility (z > 0)
Verification: Project known 3D points and overlay them on rendered images.

4.3 Part 3 — Space Carving
1. Define a 3D voxel grid enclosing the object
2. For each voxel:
• Project into each camera view
• Check if it lies inside the silhouette
3. Remove voxels that violate silhouette consistency
Output: Binary occupancy grid representing the visual hull.
4.4 Part 4 — Visualization of the Visual Hull
Convert the carved volume into a geometric representation:
• Point cloud (voxel centers), or
• Intermediate mesh (optional)
Display results using PyTorch3D, Open3D, or another tool.
4.5 Part 5 — Mesh Reconstruction (Open3D)
1. Convert the visual hull into an Open3D point cloud
2. Estimate normals
3. Apply surface reconstruction:
• Poisson reconstruction, or
• Ball pivoting
4. Visualize the reconstructed mesh

5 Experiments
Analyze:
• Effect of number of views (e.g., 5, 10, 30, 50)
• Effect of voxel resolution
• Reconstruction limitations:
– Missing concavities
– Sensitivity to silhouette quality

6 Deliverables
6.1 Code
• Clean, modular implementation
• May be submitted as:
– Jupyter notebook
– Python scripts
6.2 Report
The report may be:
• Included in the Jupyter notebook, or
• Submitted as a separate PDF
Include:
• Method description
• Key equations
• Visual results
• Analysis of experiments
6.3 Results
Include:
• Silhouettes
• Visual hull reconstructions
• Final meshes
• Comparisons across settings

7 Notes
• Ensure consistency between PyTorch3D cameras and projection implementation
• Pay attention to coordinate conventions, indexing, and scaling
• MeshLab can be used for validation and comparison of reconstruction results

8 Expected Pitfalls
• Camera convention mismatch between rendering and projection
• Incorrect use of intrinsics (pixel vs normalized coordinates)
• Row vs column indexing confusion
• Improper voxel grid bounds or scaling
• Noisy or inaccurate silhouettes
• Missing visibility check (z > 0)
• Slow performance due to non-vectorized implementation

9 Grading Criteria
Rendering + data setup 20%
Projection correctness 20%
Space carving algorithm 25%
Mesh reconstruction 15%
Visualization 5%
Report & analysis 15%