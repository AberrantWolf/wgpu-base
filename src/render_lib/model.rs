use std::ops::Range;
use std::path::Path;
use super::texture;
use tobj;

pub trait Vertex {
     const ATTRIBUTES: [wgpu::VertexAttribute; 5] = wgpu::vertex_attr_array![
            0 => Float32x3, // position
            1 => Float32x2, // tex_coords
            2 => Float32x3, // normal
            3 => Float32x3, // tangent
            4 => Float32x3, // bitangent
        ];

    fn desc() -> wgpu::VertexBufferLayout<'static>;
}

pub trait DrawModel<'a> {
    fn draw_mesh(
        &mut self,
        mesh: &'a Mesh,
        material: &'a Material,
        camera_bind_group: &'a wgpu::BindGroup,
        light_bind_group: &'a wgpu::BindGroup,
    );
    fn draw_mesh_instanced(
        &mut self,
        mesh: &'a Mesh,
        material: &'a Material,
        instances: Range<u32>,
        camera_bind_group: &'a wgpu::BindGroup,
        light_bind_group: &'a wgpu::BindGroup,
    );

    fn draw_model(
        &mut self,
        model: &'a Model,
        camera_bind_group: &'a wgpu::BindGroup,
        light_bind_group: &'a wgpu::BindGroup,
    );
    fn draw_model_instanced(
        &mut self,
        model: &'a Model,
        instances: Range<u32>,
        camera_bind_group: &'a wgpu::BindGroup,
        light_bind_group: &'a wgpu::BindGroup,
    );
}

// Add functionality to wgpu::RenderPass to draw our meshes
// Could also have been put on Mesh with a ref to &mut RenderPass
impl<'a, 'b> DrawModel<'b> for wgpu::RenderPass<'a>
where
    'b: 'a,
{
    fn draw_mesh(
        &mut self,
        mesh: &'b Mesh,
        material: &'a Material,
        camera_bind_group: &'a wgpu::BindGroup,
        light_bind_group: &'a wgpu::BindGroup,
    ) {
        self.draw_mesh_instanced(mesh, material, 0..1, camera_bind_group, light_bind_group);
    }

    fn draw_mesh_instanced(
        &mut self,
        mesh: &'b Mesh,
        material: &'b Material,
        instances: Range<u32>,
        camera_bind_group: &'b wgpu::BindGroup,
        light_bind_group: &'a wgpu::BindGroup,
    ) {
        self.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        self.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        self.set_bind_group(0, &material.bind_group, &[]);
        self.set_bind_group(1, camera_bind_group, &[]);
        self.set_bind_group(2, light_bind_group, &[]);
        self.draw_indexed(0..mesh.num_elements, 0, instances);
    }

    fn draw_model(
        &mut self,
        model: &'a Model,
        camera_bind_group: &'a wgpu::BindGroup,
        light_bind_group: &'a wgpu::BindGroup,
    ) {
        self.draw_model_instanced(model, 0..1, camera_bind_group, light_bind_group);
    }
    fn draw_model_instanced(
        &mut self,
        model: &'a Model,
        instances: Range<u32>,
        camera_bind_group: &'a wgpu::BindGroup,
        light_bind_group: &'a wgpu::BindGroup,
    ) {
        for mesh in &model.meshes {
            let material = &model.materials[mesh.material];
            self.draw_mesh_instanced(
                mesh,
                material,
                instances.clone(),
                camera_bind_group,
                light_bind_group,
            );
        }
    }
}

pub trait DrawLight<'a> {
    fn draw_light_mesh(
        &mut self,
        mesh: &'a Mesh,
        camera_bind_group: &'a wgpu::BindGroup,
        light_bind_group: &'a wgpu::BindGroup,
    );

    fn draw_light_mesh_instanced(
        &mut self,
        mesh: &'a Mesh,
        instances: Range<u32>,
        camera_bind_group: &'a wgpu::BindGroup,
        light_bind_group: &'a wgpu::BindGroup,
    );

    fn draw_light_model(
        &mut self,
        model: &'a Model,
        camera_bind_group: &'a wgpu::BindGroup,
        light_bind_group: &'a wgpu::BindGroup,
    );

    fn draw_light_model_instanced(
        &mut self,
        model: &'a Model,
        instances: Range<u32>,
        camera_bind_group: &'a wgpu::BindGroup,
        light_bind_group: &'a wgpu::BindGroup,
    );
}

impl<'a, 'b> DrawLight<'b> for wgpu::RenderPass<'a>
where
    'b: 'a,
{
    fn draw_light_mesh(
        &mut self,
        mesh: &'a Mesh,
        camera_bind_group: &'a wgpu::BindGroup,
        light_bind_group: &'a wgpu::BindGroup,
    ) {
        self.draw_light_mesh_instanced(mesh, 0..1, camera_bind_group, light_bind_group);
    }

    fn draw_light_mesh_instanced(
        &mut self,
        mesh: &'a Mesh,
        instances: Range<u32>,
        camera_bind_group: &'a wgpu::BindGroup,
        light_bind_group: &'a wgpu::BindGroup,
    ) {
        self.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        self.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        self.set_bind_group(0, camera_bind_group, &[]);
        self.set_bind_group(1, light_bind_group, &[]);
        self.draw_indexed(0..mesh.num_elements, 0, instances);
    }

    fn draw_light_model(
        &mut self,
        model: &'a Model,
        camera_bind_group: &'a wgpu::BindGroup,
        light_bind_group: &'a wgpu::BindGroup,
    ) {
        self.draw_light_model_instanced(model, 0..1, camera_bind_group, light_bind_group);
    }

    fn draw_light_model_instanced(
        &mut self,
        model: &'a Model,
        instances: Range<u32>,
        camera_bind_group: &'a wgpu::BindGroup,
        light_bind_group: &'a wgpu::BindGroup,
    ) {
        for mesh in &model.meshes {
            self.draw_light_mesh_instanced(
                mesh,
                instances.clone(),
                camera_bind_group,
                light_bind_group,
            );
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ModelVertex {
    pub position: [f32; 3],
    pub tex_coords: [f32; 2],
    pub normal: [f32; 3],
    pub tangent: [f32; 3],
    pub bitangent: [f32; 3],
}

impl Vertex for ModelVertex {
   fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<ModelVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES, // Use the const in the trait so you can share it amongst other impls..
        }
    }
}

pub struct Material {
    pub name: String,
    pub diffuse_texture: texture::Texture,
    pub normal_texture: texture::Texture,
    pub bind_group: wgpu::BindGroup,
}

impl Material {
    pub fn new(
        device: &wgpu::Device,
        name: &str,
        diffuse_texture: texture::Texture,
        normal_texture: texture::Texture,
        layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&diffuse_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&diffuse_texture.sampler),
                },
                // NEW!
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&normal_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&normal_texture.sampler),
                },
            ],
            label: Some(name),
        });

        Self {
            name: String::from(name),
            diffuse_texture,
            normal_texture,
            bind_group,
        }
    }
}

pub struct Mesh {
    pub name: String,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub num_elements: u32,
    pub material: usize,
}

pub struct Model {
    pub meshes: Vec<Mesh>,
    pub materials: Vec<Material>,
}

impl Model {
    /// Load and upload a model directly to the GPU from file
    pub async fn load_from_file<P: AsRef<Path> + std::fmt::Debug>(
        file_name: P,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
    ) -> Result<Self, crate::error::WgpuBaseError> {
        use crate::assets;
        assets::load_model(file_name, device, queue, layout).await
    }
    
    /// Create a model from raw parsed data and upload to GPU
    pub fn from_raw_data(
        raw_data: crate::model::RawModelData,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture_bind_group_layout: &wgpu::BindGroupLayout,
        base_path: &std::path::Path,
    ) -> Result<Self, crate::error::WgpuBaseError> {
        use super::texture;
        use std::path::Path;
        use wgpu::util::DeviceExt;
        
        let mut materials = Vec::new();
        for m in raw_data.materials {
            let diffuse_texture = if let Some(diffuse_path) = &m.diffuse_texture {
                let diffuse_tex_path = base_path.join(diffuse_path);
                // For now, we'll need to load the texture from file
                // This would require async, so we skip for now or we could create a default
                // For this implementation, we'll create a default texture
                texture::Texture::from_bytes(
                    device,
                    queue,
                    &image::RgbaImage::new(1, 1).as_raw(),
                    "default_diffuse",
                    false,
                )?
            } else {
                texture::Texture::from_bytes(
                    device,
                    queue,
                    &image::RgbaImage::new(1, 1).as_raw(),
                    "default_diffuse",
                    false,
                )?
            };

            let normal_texture = if let Some(normal_path) = &m.normal_texture {
                let normal_tex_path = base_path.join(normal_path);
                // Similar to diffuse, we'd load from file but for now use default
                texture::Texture::from_bytes(
                    device,
                    queue,
                    &vec![128u8, 128u8, 255u8, 255u8], // typical normal map default color
                    "default_normal",
                    true,
                )?
            } else {
                texture::Texture::from_bytes(
                    device,
                    queue,
                    &vec![128u8, 128u8, 255u8, 255u8], // typical normal map default color
                    "default_normal",
                    true,
                )?
            };

            materials.push(Material::new(
                device,
                &m.name,
                diffuse_texture,
                normal_texture,
                texture_bind_group_layout,
            ));
        }

        let meshes = raw_data
            .models
            .into_iter()
            .map(|tobj_model| {
                let m = tobj_model.mesh;
                let mut vertices = (0..m.positions.len() / 3)
                    .map(|i| {
                        let normals = if m.normals.is_empty() {
                            [0.0, 0.0, 0.0]
                        } else {
                            [
                                m.normals[i * 3],
                                m.normals[i * 3 + 1],
                                m.normals[i * 3 + 2],
                            ]
                        };

                        ModelVertex {
                            position: [
                                m.positions[i * 3],
                                m.positions[i * 3 + 1],
                                m.positions[i * 3 + 2],
                            ],
                            tex_coords: [m.texcoords[i * 2], 1.0 - m.texcoords[i * 2 + 1]],
                            normal: normals,
                            tangent: [0.0; 3],
                            bitangent: [0.0; 3],
                        }
                    })
                    .collect::<Vec<_>>();

                let indices = &m.indices;
                let mut triangles_included = vec![0; vertices.len()];

                for c in indices.chunks(3) {
                    let v0 = vertices[c[0] as usize];
                    let v1 = vertices[c[1] as usize];
                    let v2 = vertices[c[2] as usize];

                    let pos0: cgmath::Vector3<_> = v0.position.into();
                    let pos1: cgmath::Vector3<_> = v1.position.into();
                    let pos2: cgmath::Vector3<_> = v2.position.into();

                    let uv0: cgmath::Vector2<_> = v0.tex_coords.into();
                    let uv1: cgmath::Vector2<_> = v1.tex_coords.into();
                    let uv2: cgmath::Vector2<_> = v2.tex_coords.into();

                    let delta_pos1 = pos1 - pos0;
                    let delta_pos2 = pos2 - pos0;

                    let delta_uv1 = uv1 - uv0;
                    let delta_uv2 = uv2 - uv0;

                    let r = 1.0 / (delta_uv1.x * delta_uv2.y - delta_uv1.y * delta_uv2.x);
                    let tangent = (delta_pos1 * delta_uv2.y - delta_pos2 * delta_uv1.y) * r;
                    // We flip the bitangent to enable right-handed normal
                    // maps with wgpu texture coordinate system
                    let bitangent = (delta_pos2 * delta_uv1.x - delta_pos1 * delta_uv2.x) * -r;

                    // We'll use the same tangent/bitangent for each vertex in the triangle
                    vertices[c[0] as usize].tangent =
                        (tangent + cgmath::Vector3::from(vertices[c[0] as usize].tangent)).into();
                    vertices[c[1] as usize].tangent =
                        (tangent + cgmath::Vector3::from(vertices[c[1] as usize].tangent)).into();
                    vertices[c[2] as usize].tangent =
                        (tangent + cgmath::Vector3::from(vertices[c[2] as usize].tangent)).into();
                    vertices[c[0] as usize].bitangent =
                        (bitangent + cgmath::Vector3::from(vertices[c[0] as usize].bitangent)).into();
                    vertices[c[1] as usize].bitangent =
                        (bitangent + cgmath::Vector3::from(vertices[c[1] as usize].bitangent)).into();
                    vertices[c[2] as usize].bitangent =
                        (bitangent + cgmath::Vector3::from(vertices[c[2] as usize].bitangent)).into();

                    triangles_included[c[0] as usize] += 1;
                    triangles_included[c[1] as usize] += 1;
                    triangles_included[c[2] as usize] += 1;
                }

                // We have to average all the tangents/bitangents for each vertex since we accumulated them above
                for (i, n) in triangles_included.into_iter().enumerate() {
                    let denom = 1.0 / n as f32;
                    let v = &mut vertices[i];
                    v.tangent = (cgmath::Vector3::from(v.tangent) * denom).into();
                    v.bitangent = (cgmath::Vector3::from(v.bitangent) * denom).into();
                }

                let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("{:?} Vertex Buffer", "raw_model")),
                    contents: bytemuck::cast_slice(&vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                });

                let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("{:?} Index Buffer", "raw_model")),
                    contents: bytemuck::cast_slice(indices),
                    usage: wgpu::BufferUsages::INDEX,
                });

                vec![Mesh {
                    name: "raw_model".to_string(),
                    vertex_buffer,
                    index_buffer,
                    num_elements: m.indices.len() as u32,
                    material: m.material_id.unwrap_or(0),
                }]
            })
            .collect::<Vec<_>>();

        Ok(Model { meshes, materials })
    }
}

// Raw model data structure for CPU-side processing
pub struct RawModelData {
    pub models: Vec<tobj::Model>,
    pub materials: Vec<tobj::Material>,
}

impl RawModelData {
    /// Load model data from a string (e.g., OBJ file content)
    pub fn from_str(obj_content: &str) -> Result<Self, crate::error::WgpuBaseError> {
        use std::io::{BufReader, Cursor};
        
        let obj_cursor = Cursor::new(obj_content);
        let mut obj_reader = BufReader::new(obj_cursor);

        // We can't use the callback in from_str since it would require loading external files
        // So we'll return the raw data that can be processed later
        let (models, obj_materials) = tobj::load_obj_buf(
            &mut obj_reader,
            &tobj::LoadOptions {
                triangulate: true,
                single_index: true,
                ..Default::default()
            },
            |_| {
                // We don't load materials here since we don't have file paths
                // Materials will be loaded separately when uploading to GPU
                tobj::load_mtl_buf(&mut BufReader::new(Cursor::new("")))
            },
        )?;

        Ok(RawModelData {
            models,
            materials: obj_materials?,
        })
    }
}
