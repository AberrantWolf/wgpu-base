use thiserror::Error;

#[derive(Error, Debug)]
pub enum WgpuBaseError {
    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Wgpu(#[from] wgpu::Error),

    #[error(transparent)]
    RequestAdapter(#[from] wgpu::RequestAdapterError),

    #[error(transparent)]
    RequestDevice(#[from] wgpu::RequestDeviceError),

    #[error(transparent)]
    Winit(#[from] winit::error::EventLoopError),

    #[error(transparent)]
    Model(#[from] tobj::LoadError),

    #[error(transparent)]
    Image(#[from] image::ImageError),

    #[error("Texture error: {0}")]
    Texture(String),

    #[error("Asset loading error: {0}")]
    Asset(String),

    #[error("Window creation failed")]
    WindowCreationFailed,

    #[error("Surface creation failed")]
    SurfaceCreationFailed,

    #[error("Asset not found: {path}")]
    AssetNotFound { path: String },

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}