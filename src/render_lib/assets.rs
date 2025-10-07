use super::{error::WgpuBaseError, model, texture};
use std::{
    io::{self, BufReader, Cursor},
    path::Path,
};
use wgpu::util::DeviceExt;

pub fn load_string<P:AsRef<Path>>(file_name:P) -> Result<String, WgpuBaseError> {
    let txt = {
        let path = std::path::Path::new(env!("OUT_DIR"))
            .join("assets")
            .join(file_name);
        log::debug!("Loading file to string: {}", path.display());
        std::fs::read_to_string(path)?
    };
    Ok(txt)
}

pub async fn load_binary<P:AsRef<Path>>(file_name: P) -> Result<Vec<u8>, WgpuBaseError> {
    let data = {
        let path = std::path::Path::new(env!("OUT_DIR"))
            .join("assets")
            .join(file_name);
        log::debug!("Loading file to string: {}", path.display());
        std::fs::read(path)?
    };

    Ok(data)
}

pub async fn load_texture<P: AsRef<Path>>(
    file_name: P,
    is_normal_map: bool,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Result<texture::Texture, WgpuBaseError> {
    let path = file_name.as_ref();
    let data = load_binary(path).await?;
    let label = path.file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("texture");
    texture::Texture::from_bytes(device, queue, &data, label, is_normal_map)
}

pub async fn load_model<P: AsRef<Path> + std::fmt::Debug>(
    file_name: P,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
) -> Result<model::Model, WgpuBaseError> {
    let path = file_name.as_ref();
    let base_path = path.parent().unwrap_or(Path::new(""));
    let obj_text = load_string(path)?;
    let obj_cursor = Cursor::new(obj_text);
    let mut obj_reader = BufReader::new(obj_cursor);

    let (models, obj_materials) = tobj::load_obj_buf(
        &mut obj_reader,
        &tobj::LoadOptions {
            triangulate: true,
            single_index: true,
            ..Default::default()
        },
        |p| {
            let full_path = base_path.join(p);
            let mat_text = match std::fs::read_to_string(full_path)?;
            tobj::load_mtl_buf(&mut BufReader::new(Cursor::new(mat_text)))
        },
    )?;

    let mut materials = Vec::new();
    for m in obj_materials? {
        let diffuse_texture = if let Some(diffuse_path) = &m.diffuse_texture {
            let diffuse_tex_path = base_path.join(diffuse_path);
            load_texture(&diffuse_tex_path, false, device, queue).await?
        } else {
            // Create a default white texture as a fallback
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
            load_texture(&normal_tex_path, true, device, queue).await?
        } else {
            // Create a default normal texture (blue color representing normal = [0,0,1])
            texture::Texture::from_bytes(
                device,
                queue,
                &vec![128u8, 128u8, 255u8, 255u8], // typical normal map default color
                "default_normal",
                true,
            )?
        };

        materials.push(model::Material::new(
            device,
            &m.name,
            diffuse_texture,
            normal_texture,
            layout,
        ));
    }

    let meshes = models
        .into_iter()
        .map(|m| {
            let mut vertices = (0..m.mesh.positions.len() / 3)
                .map(|i| {
                    let normals = if m.mesh.normals.is_empty() {
                        [0.0, 0.0, 0.0]
                    } else {
                        [
                            m.mesh.normals[i * 3],
                            m.mesh.normals[i * 3 + 1],
                            m.mesh.normals[i * 3 + 2],
                        ]
                    };

                    model::ModelVertex {
                        position: [
                            m.mesh.positions[i * 3],
                            m.mesh.positions[i * 3 + 1],
                            m.mesh.positions[i * 3 + 2],
                        ],
                        tex_coords: [m.mesh.texcoords[i * 2], 1.0 - m.mesh.texcoords[i * 2 + 1]],
                        normal: normals,
                        tangent: [0.0; 3],
                        bitangent: [0.0; 3],
                    }
                })
                .collect::<Vec<_>>();

            let indices = &m.mesh.indices;
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
                label: Some(&format!("{:?} Vertex Buffer", file_name)),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });

            let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{:?} Index Buffer", file_name)),
                contents: bytemuck::cast_slice(indices),
                usage: wgpu::BufferUsages::INDEX,
            });

            model::Mesh {
                name: path.display().to_string(),
                vertex_buffer,
                index_buffer,
                num_elements: m.mesh.indices.len() as u32,
                material: m.mesh.material_id.unwrap_or(0),
            }
        })
        .collect::<Vec<_>>();

    Ok(model::Model { meshes, materials })
}
