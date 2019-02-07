use ocl::builders::ContextProperties;
use ocl::core;

pub fn boilerplate() -> ocl::core::Result<(
    ocl::core::PlatformId,
    ocl::core::DeviceId,
    ocl::core::ContextProperties,
    ocl::core::Context,
)> {
    let platform = core::default_platform()?;
    let devices = core::get_device_ids(&platform, None, None)?;
    let device = devices[0];
    let context_properties = ContextProperties::new().platform(platform);
    let context = core::create_context(Some(&context_properties), &[device], None, None)?;
    Ok((platform, device, context_properties, context))
}
