//! Headless GPU access: instance, adapter, device and queue acquisition
//! without a window or surface.

/// A connected GPU: everything needed to create resources and submit work.
pub struct Gpu {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

/// Why [`Gpu::headless`] failed.
#[derive(Debug)]
pub enum GpuError {
    /// No adapter matched the request options on this machine.
    NoAdapter,
    /// An adapter was found but the device could not be created.
    RequestDevice(String),
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuError::NoAdapter => write!(f, "no suitable GPU adapter found"),
            GpuError::RequestDevice(msg) => write!(f, "requesting device failed: {msg}"),
        }
    }
}

impl std::error::Error for GpuError {}

impl Gpu {
    /// Creates a GPU connection with no surface attached: all backends,
    /// high-performance adapter preference, fallback adapters allowed.
    /// Returns [`GpuError::NoAdapter`] when the machine has no usable
    /// adapter. Never panics.
    pub fn headless() -> Result<Gpu, GpuError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        Self::connect(instance, None)
    }

    /// Creates a GPU connection that can present to `surface`. The
    /// instance must be the one the surface was created from.
    pub fn for_surface(
        instance: wgpu::Instance,
        surface: &wgpu::Surface<'_>,
    ) -> Result<Gpu, GpuError> {
        Self::connect(instance, Some(surface))
    }

    fn connect(
        instance: wgpu::Instance,
        compatible_surface: Option<&wgpu::Surface<'_>>,
    ) -> Result<Gpu, GpuError> {
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface,
            force_fallback_adapter: false,
            apply_limit_buckets: Default::default(),
        }))
        .map_err(|_| GpuError::NoAdapter)?;
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                .map_err(|e| GpuError::RequestDevice(e.to_string()))?;
        Ok(Gpu {
            instance,
            adapter,
            device,
            queue,
        })
    }
}
