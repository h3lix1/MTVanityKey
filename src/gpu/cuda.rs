use std::ffi::{CStr, CString};
use std::mem;
use std::os::raw::{c_char, c_int, c_uint, c_void};
use std::ptr;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use libloading::Library;

use super::{ed_step_table, Match, MeshSearcher, SearchParams, Searcher};
use crate::prefix::Prefix;
use crate::table;

/// The fatbin build.rs compiled from src/kernels/ed25519.cu.
const CUDA_MODULE_IMAGE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/mtvanitykey_cuda.cubin"));
const CUDA_SUCCESS: CUresult = 0;

type CUdevice = c_int;
type CUcontext = *mut c_void;
type CUdeviceptr = u64;
type CUfunction = *mut c_void;
type CUmodule = *mut c_void;
type CUresult = c_int;
type CUstream = *mut c_void;

type CuInit = unsafe extern "C" fn(c_uint) -> CUresult;
type CuDeviceGetCount = unsafe extern "C" fn(*mut c_int) -> CUresult;
type CuDeviceGet = unsafe extern "C" fn(*mut CUdevice, c_int) -> CUresult;
type CuDeviceGetAttribute = unsafe extern "C" fn(*mut c_int, c_int, CUdevice) -> CUresult;
type CuDeviceGetName = unsafe extern "C" fn(*mut c_char, c_int, CUdevice) -> CUresult;
type CuCtxCreate = unsafe extern "C" fn(*mut CUcontext, c_uint, CUdevice) -> CUresult;
type CuCtxDestroy = unsafe extern "C" fn(CUcontext) -> CUresult;
type CuCtxSetCurrent = unsafe extern "C" fn(CUcontext) -> CUresult;
type CuCtxSynchronize = unsafe extern "C" fn() -> CUresult;
type CuModuleLoadData = unsafe extern "C" fn(*mut CUmodule, *const c_void) -> CUresult;
type CuModuleUnload = unsafe extern "C" fn(CUmodule) -> CUresult;
type CuModuleGetFunction =
    unsafe extern "C" fn(*mut CUfunction, CUmodule, *const c_char) -> CUresult;
type CuMemAlloc = unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUresult;
type CuMemFree = unsafe extern "C" fn(CUdeviceptr) -> CUresult;
type CuMemcpyHtoD = unsafe extern "C" fn(CUdeviceptr, *const c_void, usize) -> CUresult;
type CuLaunchKernel = unsafe extern "C" fn(
    CUfunction,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    CUstream,
    *mut *mut c_void,
    *mut *mut c_void,
) -> CUresult;
type CuGetErrorName = unsafe extern "C" fn(CUresult, *mut *const c_char) -> CUresult;
type CuGetErrorString = unsafe extern "C" fn(CUresult, *mut *const c_char) -> CUresult;
type CUevent = *mut c_void;
type CuMemHostAlloc = unsafe extern "C" fn(*mut *mut c_void, usize, c_uint) -> CUresult;
type CuMemFreeHost = unsafe extern "C" fn(*mut c_void) -> CUresult;
type CuMemcpyHtoDAsync = unsafe extern "C" fn(CUdeviceptr, *const c_void, usize, CUstream) -> CUresult;
type CuMemcpyDtoHAsync = unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize, CUstream) -> CUresult;
type CuMemsetD32Async = unsafe extern "C" fn(CUdeviceptr, c_uint, usize, CUstream) -> CUresult;
type CuStreamCreate = unsafe extern "C" fn(*mut CUstream, c_uint) -> CUresult;
type CuStreamDestroy = unsafe extern "C" fn(CUstream) -> CUresult;
type CuEventCreate = unsafe extern "C" fn(*mut CUevent, c_uint) -> CUresult;
type CuEventDestroy = unsafe extern "C" fn(CUevent) -> CUresult;
type CuEventRecord = unsafe extern "C" fn(CUevent, CUstream) -> CUresult;
type CuEventSynchronize = unsafe extern "C" fn(CUevent) -> CUresult;

/// cuEventCreate flags: the waiting host thread sleeps instead of spinning
/// (the CPU workers want that core), and no timing bookkeeping.
const CU_EVENT_BLOCKING_SYNC: c_uint = 0x1;
const CU_EVENT_DISABLE_TIMING: c_uint = 0x2;

/// Result block layout shared by one launch slot (u32 words):
/// [found, found_index, found_iter, debug_prefix[8]].
const SEARCH_RESULT_WORDS: usize = 11;

/// Launch slots per pipelined backend (= `pipeline_depth()`): one batch runs
/// on the GPU while the host fills, uploads and queues the next.
const PIPELINE_SLOTS: usize = 2;

pub(super) struct CudaSearcher {
    params_buffer: DeviceBuffer,
    basepoint_table_buffer: DeviceBuffer,
    /// Affine step multiples 1*S..window*S for the window walk.
    step_table_buffer: DeviceBuffer,
    /// Batch N runs while the host prepares batch N+1.
    slots: [LaunchSlot; PIPELINE_SLOTS],
    stream: CudaStream,
    /// Number of batches submitted so far; slot = seq % PIPELINE_SLOTS.
    submitted: usize,
    /// Number of batches whose results have been collected.
    collected: usize,
    _module: CudaModule,
    context: CudaContext,
    driver: Arc<CudaDriver>,
    function: CUfunction,
    batch_size: usize,
    iters: u32,
    /// The full 32-byte compressed point (y encoding plus x sign bit) the
    /// kernel computed for the first candidate of gid 0 on the last
    /// collected launch, as eight little-endian u32 words.
    last_debug_point: [u32; 8],
}

impl CudaSearcher {
    pub(super) fn new(batch_size: usize, iters: u32, prefix: &Prefix) -> Result<Self> {
        let driver = Arc::new(CudaDriver::load()?);
        unsafe {
            driver.check((driver.cu_init)(0), "cuInit")?;
        }

        let context = CudaContext::create(Arc::clone(&driver))?;
        let module = CudaModule::load(Arc::clone(&driver), context.raw())?;
        let function = module.function("search_kernel")?;
        let params = SearchParams::new(batch_size, iters, prefix);

        let params_buffer = DeviceBuffer::with_value(Arc::clone(&driver), context.raw(), &params)?;
        let basepoint_table_buffer =
            DeviceBuffer::with_slice(Arc::clone(&driver), context.raw(), table::basepoint_table())?;
        let step_table_buffer = DeviceBuffer::with_slice(
            Arc::clone(&driver),
            context.raw(),
            &ed_step_table::step_table(batch_inv_window()),
        )?;
        let slots = [
            LaunchSlot::new(&driver, context.raw(), batch_size * 8, SEARCH_RESULT_WORDS)?,
            LaunchSlot::new(&driver, context.raw(), batch_size * 8, SEARCH_RESULT_WORDS)?,
        ];
        let stream = CudaStream::new(Arc::clone(&driver), context.raw())?;

        Ok(Self {
            params_buffer,
            basepoint_table_buffer,
            step_table_buffer,
            slots,
            stream,
            submitted: 0,
            collected: 0,
            _module: module,
            context,
            driver,
            function,
            batch_size,
            iters,
            last_debug_point: [0u32; 8],
        })
    }

    /// Queue one batch on the stream (upload, zero results, launch, download).
    fn enqueue(&mut self, words: &[u32]) -> Result<()> {
        if words.len() != self.batch_size * 8 {
            return Err(anyhow!(
                "expected {} scalar words, got {}",
                self.batch_size * 8,
                words.len()
            ));
        }
        if self.submitted - self.collected >= PIPELINE_SLOTS {
            return Err(anyhow!("CUDA pipeline overfilled: collect a batch before submitting"));
        }
        self.context.set_current()?;
        let slot_index = self.submitted % PIPELINE_SLOTS;
        let block_size = block_size();
        let grid_size = (self.batch_size as u32).div_ceil(block_size);
        let function = self.function;
        let driver = Arc::clone(&self.driver);
        let mut params = self.params_buffer.ptr;
        let mut basepoint_table = self.basepoint_table_buffer.ptr;
        let mut step_table = self.step_table_buffer.ptr;
        let slot = &mut self.slots[slot_index];
        slot.enqueue(words, &self.stream, |scalars_dev, results_dev, stream| {
            let mut scalars = scalars_dev.ptr;
            let mut found = results_dev.ptr;
            let mut found_index = results_dev.ptr + 4;
            let mut found_iter = results_dev.ptr + 8;
            let mut debug_prefix = results_dev.ptr + 12;
            let mut kernel_args = [
                &mut scalars as *mut CUdeviceptr as *mut c_void,
                &mut params as *mut CUdeviceptr as *mut c_void,
                &mut found as *mut CUdeviceptr as *mut c_void,
                &mut found_index as *mut CUdeviceptr as *mut c_void,
                &mut found_iter as *mut CUdeviceptr as *mut c_void,
                &mut debug_prefix as *mut CUdeviceptr as *mut c_void,
                &mut basepoint_table as *mut CUdeviceptr as *mut c_void,
                &mut step_table as *mut CUdeviceptr as *mut c_void,
            ];
            driver.launch_on(function, grid_size, block_size, kernel_args.as_mut_ptr(), stream.raw)
        })?;
        self.submitted += 1;
        Ok(())
    }

    /// Wait for the oldest in-flight batch and decode its result block.
    fn collect_oldest(&mut self) -> Result<Option<Match>> {
        if self.collected >= self.submitted {
            return Ok(None);
        }
        let slot = &mut self.slots[self.collected % PIPELINE_SLOTS];
        slot.wait()?;
        self.collected += 1;
        let results = slot.results.host.as_slice();
        self.last_debug_point.copy_from_slice(&results[3..11]);
        if results[0] == 0 {
            return Ok(None);
        }
        let index = results[1] as usize;
        if index >= self.batch_size {
            return Err(anyhow!("CUDA returned out-of-range index {index}"));
        }
        let iter = results[2];
        if iter >= self.iters {
            return Err(anyhow!("CUDA returned out-of-range iteration {iter}"));
        }
        Ok(Some(Match { index, iter }))
    }

}

impl Drop for CudaSearcher {
    /// Let any still-queued batch (an error path or an early stop) finish
    /// before the pinned/device buffers it writes are freed field by field.
    fn drop(&mut self) {
        if self.context.set_current().is_ok() {
            let _ = self.driver.synchronize();
        }
    }
}

impl Searcher for CudaSearcher {
    fn search_words(&mut self, words: &[u32]) -> Result<Option<Match>> {
        if self.collected != self.submitted {
            return Err(anyhow!("search_words called with a pipelined batch in flight"));
        }
        self.enqueue(words)?;
        self.collect_oldest()
    }

    /// Two launch slots: batch N runs on the GPU while the host fills and
    /// uploads batch N+1 (the worker loop keeps `depth` launches queued and
    /// collects the oldest before refilling its slot).
    fn pipeline_depth(&self) -> usize {
        PIPELINE_SLOTS
    }

    fn submit(&mut self, words: &[u32]) -> Result<()> {
        self.enqueue(words)
    }

    fn collect(&mut self) -> Result<Option<Match>> {
        if self.collected >= self.submitted {
            return Err(anyhow!("collect called with no CUDA batch in flight"));
        }
        self.collect_oldest()
    }

    fn last_debug_point(&self) -> [u8; 32] {
        let mut bytes = [0u8; 32];
        for (chunk, word) in bytes.chunks_exact_mut(4).zip(self.last_debug_point) {
            chunk.copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }
}

impl CudaDriver {
    /// cuInit + error check.
    fn initialize(&self) -> Result<()> {
        unsafe {
            self.check((self.cu_init)(0), "cuInit")?;
        }
        Ok(())
    }

    /// Launch a kernel with a flat grid/block on `stream` (the kernels use
    /// static shared memory only).
    fn launch_on(
        &self,
        function: CUfunction,
        grid: u32,
        block: u32,
        args: *mut *mut c_void,
        stream: CUstream,
    ) -> Result<()> {
        unsafe {
            self.check(
                (self.cu_launch_kernel)(function, grid, 1, 1, block, 1, 1, 0, stream, args, ptr::null_mut()),
                "cuLaunchKernel",
            )
        }
    }

    /// Device-0 attribute, or an error.
    fn device_attribute(&self, attribute: c_int) -> Result<c_int> {
        let mut device = 0;
        let mut value = 0;
        unsafe {
            self.check((self.cu_device_get)(&mut device, 0), "cuDeviceGet")?;
            self.check((self.cu_device_get_attribute)(&mut value, attribute, device), "cuDeviceGetAttribute")?;
        }
        Ok(value)
    }

    fn synchronize(&self) -> Result<()> {
        unsafe { self.check((self.cu_ctx_synchronize)(), "cuCtxSynchronize") }
    }
}

struct CudaDriver {
    _library: Library,
    cu_init: CuInit,
    cu_device_get_count: CuDeviceGetCount,
    cu_device_get: CuDeviceGet,
    cu_device_get_attribute: CuDeviceGetAttribute,
    cu_device_get_name: CuDeviceGetName,
    cu_ctx_create: CuCtxCreate,
    cu_ctx_destroy: CuCtxDestroy,
    cu_ctx_set_current: CuCtxSetCurrent,
    cu_ctx_synchronize: CuCtxSynchronize,
    cu_module_load_data: CuModuleLoadData,
    cu_module_unload: CuModuleUnload,
    cu_module_get_function: CuModuleGetFunction,
    cu_mem_alloc: CuMemAlloc,
    cu_mem_free: CuMemFree,
    cu_memcpy_htod: CuMemcpyHtoD,
    cu_launch_kernel: CuLaunchKernel,
    cu_get_error_name: CuGetErrorName,
    cu_get_error_string: CuGetErrorString,
    cu_mem_host_alloc: CuMemHostAlloc,
    cu_mem_free_host: CuMemFreeHost,
    cu_memcpy_htod_async: CuMemcpyHtoDAsync,
    cu_memcpy_dtoh_async: CuMemcpyDtoHAsync,
    cu_memset_d32_async: CuMemsetD32Async,
    cu_stream_create: CuStreamCreate,
    cu_stream_destroy: CuStreamDestroy,
    cu_event_create: CuEventCreate,
    cu_event_destroy: CuEventDestroy,
    cu_event_record: CuEventRecord,
    cu_event_synchronize: CuEventSynchronize,
}

impl CudaDriver {
    fn load() -> Result<Self> {
        let library = load_cuda_library()?;
        unsafe {
            Ok(Self {
                cu_init: load_symbol(&library, b"cuInit\0")?,
                cu_device_get_count: load_symbol(&library, b"cuDeviceGetCount\0")?,
                cu_device_get: load_symbol(&library, b"cuDeviceGet\0")?,
                cu_device_get_attribute: load_symbol(&library, b"cuDeviceGetAttribute\0")?,
                cu_device_get_name: load_symbol(&library, b"cuDeviceGetName\0")?,
                cu_ctx_create: load_symbol(&library, b"cuCtxCreate_v2\0")?,
                cu_ctx_destroy: load_symbol(&library, b"cuCtxDestroy_v2\0")?,
                cu_ctx_set_current: load_symbol(&library, b"cuCtxSetCurrent\0")?,
                cu_ctx_synchronize: load_symbol(&library, b"cuCtxSynchronize\0")?,
                cu_module_load_data: load_symbol(&library, b"cuModuleLoadData\0")?,
                cu_module_unload: load_symbol(&library, b"cuModuleUnload\0")?,
                cu_module_get_function: load_symbol(&library, b"cuModuleGetFunction\0")?,
                cu_mem_alloc: load_symbol(&library, b"cuMemAlloc_v2\0")?,
                cu_mem_free: load_symbol(&library, b"cuMemFree_v2\0")?,
                cu_memcpy_htod: load_symbol(&library, b"cuMemcpyHtoD_v2\0")?,
                cu_launch_kernel: load_symbol(&library, b"cuLaunchKernel\0")?,
                cu_get_error_name: load_symbol(&library, b"cuGetErrorName\0")?,
                cu_get_error_string: load_symbol(&library, b"cuGetErrorString\0")?,
                cu_mem_host_alloc: load_symbol(&library, b"cuMemHostAlloc\0")?,
                cu_mem_free_host: load_symbol(&library, b"cuMemFreeHost\0")?,
                cu_memcpy_htod_async: load_symbol(&library, b"cuMemcpyHtoDAsync_v2\0")?,
                cu_memcpy_dtoh_async: load_symbol(&library, b"cuMemcpyDtoHAsync_v2\0")?,
                cu_memset_d32_async: load_symbol(&library, b"cuMemsetD32Async\0")?,
                cu_stream_create: load_symbol(&library, b"cuStreamCreate\0")?,
                cu_stream_destroy: load_symbol(&library, b"cuStreamDestroy_v2\0")?,
                cu_event_create: load_symbol(&library, b"cuEventCreate\0")?,
                cu_event_destroy: load_symbol(&library, b"cuEventDestroy_v2\0")?,
                cu_event_record: load_symbol(&library, b"cuEventRecord\0")?,
                cu_event_synchronize: load_symbol(&library, b"cuEventSynchronize\0")?,
                _library: library,
            })
        }
    }

    fn check(&self, result: CUresult, operation: &str) -> Result<()> {
        if result == CUDA_SUCCESS {
            Ok(())
        } else {
            Err(anyhow!(
                "{operation} failed: {}",
                self.error_message(result)
            ))
        }
    }

    fn error_message(&self, result: CUresult) -> String {
        unsafe {
            let mut name = ptr::null();
            let mut text = ptr::null();
            let has_name =
                (self.cu_get_error_name)(result, &mut name) == CUDA_SUCCESS && !name.is_null();
            let has_text =
                (self.cu_get_error_string)(result, &mut text) == CUDA_SUCCESS && !text.is_null();

            match (has_name, has_text) {
                (true, true) => format!(
                    "{} ({})",
                    CStr::from_ptr(name).to_string_lossy(),
                    CStr::from_ptr(text).to_string_lossy()
                ),
                (true, false) => CStr::from_ptr(name).to_string_lossy().into_owned(),
                _ => format!("CUDA error code {result}"),
            }
        }
    }
}

unsafe fn load_symbol<T: Copy>(library: &Library, name: &[u8]) -> Result<T> {
    Ok(*library.get::<T>(name).with_context(|| {
        format!(
            "failed to load CUDA driver symbol {}",
            String::from_utf8_lossy(name).trim_end_matches('\0')
        )
    })?)
}

fn load_cuda_library() -> Result<Library> {
    #[cfg(target_os = "windows")]
    const NAMES: &[&str] = &["nvcuda.dll"];
    #[cfg(target_os = "macos")]
    const NAMES: &[&str] = &["libcuda.dylib", "libcuda.so.1", "libcuda.so"];
    #[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
    const NAMES: &[&str] = &["libcuda.so.1", "libcuda.so"];

    let mut attempts = Vec::new();
    for name in NAMES {
        match unsafe { Library::new(name) } {
            Ok(library) => return Ok(library),
            Err(error) => attempts.push(format!("{name}: {error}")),
        }
    }

    Err(anyhow!(
        "failed to load CUDA driver library; tried {}",
        attempts.join(", ")
    ))
}

/// Name of CUDA device 0 (cuDeviceGetName).
pub(super) fn device_name() -> Result<String> {
    let driver = CudaDriver::load()?;
    driver.initialize()?;
    let mut device = 0;
    let mut name = [0 as c_char; 256];
    unsafe {
        driver.check((driver.cu_device_get)(&mut device, 0), "cuDeviceGet")?;
        driver.check(
            (driver.cu_device_get_name)(name.as_mut_ptr(), name.len() as c_int, device),
            "cuDeviceGetName",
        )?;
        Ok(CStr::from_ptr(name.as_ptr()).to_string_lossy().into_owned())
    }
}

struct CudaContext {
    driver: Arc<CudaDriver>,
    raw: CUcontext,
}

impl CudaContext {
    fn create(driver: Arc<CudaDriver>) -> Result<Self> {
        let mut device_count = 0;
        unsafe {
            driver.check(
                (driver.cu_device_get_count)(&mut device_count),
                "cuDeviceGetCount",
            )?;
        }
        if device_count <= 0 {
            return Err(anyhow!("no CUDA devices found"));
        }

        let mut device = 0;
        unsafe {
            driver.check((driver.cu_device_get)(&mut device, 0), "cuDeviceGet")?;
        }

        // CU_CTX_LMEM_RESIZE_TO_MAX: keep the per-thread local memory pool at
        // the largest size any launch needed (the kernels keep their
        // batch-inversion window in local memory, ~15-20 KB of stack per
        // thread, and the driver may otherwise resize the pool around
        // launches). CU_CTX_SCHED_BLOCKING_SYNC: the host thread waiting on
        // an event sleeps instead of spinning on a core that the CPU workers
        // want. Neither flag changed the measured throughput on an RTX 5090.
        const CU_CTX_SCHED_BLOCKING_SYNC: c_uint = 0x04;
        const CU_CTX_LMEM_RESIZE_TO_MAX: c_uint = 0x10;
        let mut raw = ptr::null_mut();
        unsafe {
            driver.check(
                (driver.cu_ctx_create)(
                    &mut raw,
                    CU_CTX_SCHED_BLOCKING_SYNC | CU_CTX_LMEM_RESIZE_TO_MAX,
                    device,
                ),
                "cuCtxCreate",
            )?;
        }

        Ok(Self { driver, raw })
    }

    fn raw(&self) -> CUcontext {
        self.raw
    }

    fn set_current(&self) -> Result<()> {
        unsafe {
            self.driver.check(
                (self.driver.cu_ctx_set_current)(self.raw),
                "cuCtxSetCurrent",
            )
        }
    }
}

impl Drop for CudaContext {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                let _ = (self.driver.cu_ctx_set_current)(self.raw);
                let _ = (self.driver.cu_ctx_destroy)(self.raw);
            }
        }
    }
}

struct CudaModule {
    driver: Arc<CudaDriver>,
    context: CUcontext,
    raw: CUmodule,
}

impl CudaModule {
    fn load(driver: Arc<CudaDriver>, context: CUcontext) -> Result<Self> {
        let mut raw = ptr::null_mut();
        unsafe {
            driver.check((driver.cu_ctx_set_current)(context), "cuCtxSetCurrent")?;
            driver.check(
                (driver.cu_module_load_data)(&mut raw, CUDA_MODULE_IMAGE.as_ptr() as *const c_void),
                "cuModuleLoadData",
            )?;
        }

        Ok(Self {
            driver,
            context,
            raw,
        })
    }

    fn function(&self, name: &str) -> Result<CUfunction> {
        let name = CString::new(name).context("CUDA function name contained an interior NUL")?;
        let mut function = ptr::null_mut();
        unsafe {
            self.driver.check(
                (self.driver.cu_module_get_function)(&mut function, self.raw, name.as_ptr()),
                "cuModuleGetFunction",
            )?;
        }
        Ok(function)
    }
}

impl Drop for CudaModule {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                let _ = (self.driver.cu_ctx_set_current)(self.context);
                let _ = (self.driver.cu_module_unload)(self.raw);
            }
        }
    }
}

struct DeviceBuffer {
    driver: Arc<CudaDriver>,
    context: CUcontext,
    ptr: CUdeviceptr,
    bytes: usize,
}

impl DeviceBuffer {
    fn new(driver: Arc<CudaDriver>, context: CUcontext, bytes: usize) -> Result<Self> {
        let mut ptr = 0;
        unsafe {
            driver.check((driver.cu_ctx_set_current)(context), "cuCtxSetCurrent")?;
            driver.check((driver.cu_mem_alloc)(&mut ptr, bytes), "cuMemAlloc")?;
        }

        Ok(Self {
            driver,
            context,
            ptr,
            bytes,
        })
    }

    fn with_value<T: Copy>(driver: Arc<CudaDriver>, context: CUcontext, value: &T) -> Result<Self> {
        let buffer = Self::new(driver, context, mem::size_of::<T>())?;
        buffer.copy_from_value(value)?;
        Ok(buffer)
    }

    fn with_slice<T: Copy>(
        driver: Arc<CudaDriver>,
        context: CUcontext,
        values: &[T],
    ) -> Result<Self> {
        let buffer = Self::new(driver, context, mem::size_of_val(values))?;
        buffer.copy_from_slice(values)?;
        Ok(buffer)
    }

    fn copy_from_value<T: Copy>(&self, value: &T) -> Result<()> {
        self.copy_from_bytes(value as *const T as *const c_void, mem::size_of::<T>())
    }

    fn copy_from_slice<T: Copy>(&self, values: &[T]) -> Result<()> {
        self.copy_from_bytes(values.as_ptr() as *const c_void, mem::size_of_val(values))
    }

    fn copy_from_bytes(&self, src: *const c_void, bytes: usize) -> Result<()> {
        if bytes > self.bytes {
            return Err(anyhow!(
                "copy of {bytes} bytes exceeds CUDA buffer size {}",
                self.bytes
            ));
        }
        unsafe {
            self.driver.check(
                (self.driver.cu_ctx_set_current)(self.context),
                "cuCtxSetCurrent",
            )?;
            self.driver.check(
                (self.driver.cu_memcpy_htod)(self.ptr, src, bytes),
                "cuMemcpyHtoD",
            )
        }
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if self.ptr != 0 {
            unsafe {
                let _ = (self.driver.cu_ctx_set_current)(self.context);
                let _ = (self.driver.cu_mem_free)(self.ptr);
            }
        }
    }
}

impl DeviceBuffer {
    /// Enqueue a host-to-device copy from pinned memory on `stream`.
    fn copy_from_pinned_async(&self, src: &PinnedBuffer, stream: &CudaStream) -> Result<()> {
        if src.bytes > self.bytes {
            return Err(anyhow!(
                "async copy of {} bytes exceeds CUDA buffer size {}",
                src.bytes,
                self.bytes
            ));
        }
        unsafe {
            self.driver.check(
                (self.driver.cu_memcpy_htod_async)(self.ptr, src.ptr, src.bytes, stream.raw),
                "cuMemcpyHtoDAsync",
            )
        }
    }

    /// Enqueue a device-to-host copy into pinned memory on `stream`.
    fn copy_to_pinned_async(&self, dst: &PinnedBuffer, stream: &CudaStream) -> Result<()> {
        if dst.bytes > self.bytes {
            return Err(anyhow!(
                "async copy of {} bytes exceeds CUDA buffer size {}",
                dst.bytes,
                self.bytes
            ));
        }
        unsafe {
            self.driver.check(
                (self.driver.cu_memcpy_dtoh_async)(dst.ptr, self.ptr, dst.bytes, stream.raw),
                "cuMemcpyDtoHAsync",
            )
        }
    }

    /// Enqueue a 32-bit fill of the whole buffer on `stream`.
    fn fill_u32_async(&self, value: u32, stream: &CudaStream) -> Result<()> {
        unsafe {
            self.driver.check(
                (self.driver.cu_memset_d32_async)(
                    self.ptr,
                    value,
                    self.bytes / mem::size_of::<u32>(),
                    stream.raw,
                ),
                "cuMemsetD32Async",
            )
        }
    }
}

/// Page-locked host memory: the only kind the driver can DMA asynchronously
/// (pageable copies are staged through a driver bounce buffer and block).
struct PinnedBuffer {
    driver: Arc<CudaDriver>,
    context: CUcontext,
    ptr: *mut c_void,
    bytes: usize,
}

impl PinnedBuffer {
    fn new_u32(driver: Arc<CudaDriver>, context: CUcontext, words: usize) -> Result<Self> {
        let bytes = words * mem::size_of::<u32>();
        let mut ptr = ptr::null_mut();
        unsafe {
            driver.check((driver.cu_ctx_set_current)(context), "cuCtxSetCurrent")?;
            driver.check((driver.cu_mem_host_alloc)(&mut ptr, bytes.max(4), 0), "cuMemHostAlloc")?;
            ptr::write_bytes(ptr as *mut u8, 0, bytes);
        }
        Ok(Self {
            driver,
            context,
            ptr,
            bytes,
        })
    }

    fn as_slice(&self) -> &[u32] {
        unsafe { std::slice::from_raw_parts(self.ptr as *const u32, self.bytes / 4) }
    }

    fn as_mut_slice(&mut self) -> &mut [u32] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut u32, self.bytes / 4) }
    }
}

impl Drop for PinnedBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                let _ = (self.driver.cu_ctx_set_current)(self.context);
                let _ = (self.driver.cu_mem_free_host)(self.ptr);
            }
        }
    }
}

struct CudaStream {
    driver: Arc<CudaDriver>,
    context: CUcontext,
    raw: CUstream,
}

impl CudaStream {
    fn new(driver: Arc<CudaDriver>, context: CUcontext) -> Result<Self> {
        let mut raw = ptr::null_mut();
        unsafe {
            driver.check((driver.cu_ctx_set_current)(context), "cuCtxSetCurrent")?;
            driver.check((driver.cu_stream_create)(&mut raw, 0), "cuStreamCreate")?;
        }
        Ok(Self {
            driver,
            context,
            raw,
        })
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                let _ = (self.driver.cu_ctx_set_current)(self.context);
                let _ = (self.driver.cu_stream_destroy)(self.raw);
            }
        }
    }
}

struct CudaEvent {
    driver: Arc<CudaDriver>,
    context: CUcontext,
    raw: CUevent,
}

impl CudaEvent {
    fn new(driver: Arc<CudaDriver>, context: CUcontext) -> Result<Self> {
        let flags = CU_EVENT_BLOCKING_SYNC | CU_EVENT_DISABLE_TIMING;
        let mut raw = ptr::null_mut();
        unsafe {
            driver.check((driver.cu_ctx_set_current)(context), "cuCtxSetCurrent")?;
            driver.check((driver.cu_event_create)(&mut raw, flags), "cuEventCreate")?;
        }
        Ok(Self {
            driver,
            context,
            raw,
        })
    }

    /// Record this event after everything queued so far on `stream`.
    fn record(&self, stream: &CudaStream) -> Result<()> {
        unsafe { self.driver.check((self.driver.cu_event_record)(self.raw, stream.raw), "cuEventRecord") }
    }

    /// Block (sleeping, not spinning) until the recorded work has completed.
    fn wait(&self) -> Result<()> {
        unsafe { self.driver.check((self.driver.cu_event_synchronize)(self.raw), "cuEventSynchronize") }
    }
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                let _ = (self.driver.cu_ctx_set_current)(self.context);
                let _ = (self.driver.cu_event_destroy)(self.raw);
            }
        }
    }
}

/// One in-flight launch's result block: a zeroed device block the kernel
/// writes into, the pinned host copy it lands in, and the event the host
/// waits on. Everything is ordered on one stream, so a slot can be queued
/// behind another slot's launch and the GPU runs them back to back.
struct ResultSlot {
    dev: DeviceBuffer,
    host: PinnedBuffer,
    event: CudaEvent,
    in_flight: bool,
}

impl ResultSlot {
    fn new(driver: &Arc<CudaDriver>, context: CUcontext, words: usize) -> Result<Self> {
        Ok(Self {
            dev: DeviceBuffer::new(Arc::clone(driver), context, words * mem::size_of::<u32>())?,
            host: PinnedBuffer::new_u32(Arc::clone(driver), context, words)?,
            event: CudaEvent::new(Arc::clone(driver), context)?,
            in_flight: false,
        })
    }

    /// Queue: zero the result block, run `launch`, download the results,
    /// record completion.
    fn enqueue(
        &mut self,
        stream: &CudaStream,
        launch: impl FnOnce(&DeviceBuffer, &CudaStream) -> Result<()>,
    ) -> Result<()> {
        if self.in_flight {
            return Err(anyhow!("CUDA launch slot reused while still in flight"));
        }
        self.dev.fill_u32_async(0, stream)?;
        launch(&self.dev, stream)?;
        self.dev.copy_to_pinned_async(&self.host, stream)?;
        self.event.record(stream)?;
        self.in_flight = true;
        Ok(())
    }

    /// Wait for the queued launch; the results are then readable in `host`.
    fn wait(&mut self) -> Result<()> {
        self.event.wait()?;
        self.in_flight = false;
        Ok(())
    }
}

/// One stage of a depth-1 launch pipeline for the per-batch-scalar kernels:
/// its own scalar upload buffers plus a result slot, so the host can fill
/// and upload the next batch while this one runs.
struct LaunchSlot {
    scalars_host: PinnedBuffer,
    scalars_dev: DeviceBuffer,
    results: ResultSlot,
}

impl LaunchSlot {
    fn new(
        driver: &Arc<CudaDriver>,
        context: CUcontext,
        scalar_words: usize,
        result_words: usize,
    ) -> Result<Self> {
        Ok(Self {
            scalars_host: PinnedBuffer::new_u32(Arc::clone(driver), context, scalar_words)?,
            scalars_dev: DeviceBuffer::new(
                Arc::clone(driver),
                context,
                scalar_words * mem::size_of::<u32>(),
            )?,
            results: ResultSlot::new(driver, context, result_words)?,
        })
    }

    /// Queue: copy `words` into the pinned staging buffer, upload them, zero
    /// the result block, run `launch`, download the results, record
    /// completion. Everything is ordered on `stream`. The in-flight check
    /// comes first so a slot whose upload DMA may still be running is never
    /// overwritten from the host.
    fn enqueue(
        &mut self,
        words: &[u32],
        stream: &CudaStream,
        launch: impl FnOnce(&DeviceBuffer, &DeviceBuffer, &CudaStream) -> Result<()>,
    ) -> Result<()> {
        if self.results.in_flight {
            return Err(anyhow!("CUDA launch slot reused while still in flight"));
        }
        if words.len() != self.scalars_host.as_slice().len() {
            return Err(anyhow!(
                "expected {} scalar words, got {}",
                self.scalars_host.as_slice().len(),
                words.len()
            ));
        }
        self.scalars_host.as_mut_slice().copy_from_slice(words);
        self.scalars_dev.copy_from_pinned_async(&self.scalars_host, stream)?;
        let scalars_dev = &self.scalars_dev;
        self.results
            .enqueue(stream, |results_dev, stream| launch(scalars_dev, results_dev, stream))
    }

    /// Wait for the queued batch; the results are then readable in `results.host`.
    fn wait(&mut self) -> Result<()> {
        self.results.wait()
    }
}

/// The compiled block size (build.rs passes -DLB_THREADS and exports it).
pub(crate) fn block_size() -> u32 {
    const BLOCK_SIZE_STR: &str = env!("MTVK_CUDA_BLOCK_SIZE");
    BLOCK_SIZE_STR.parse().expect("MTVK_CUDA_BLOCK_SIZE")
}

/// The kernels' compiled batch-inversion window (BATCH_INV_WINDOW, set by
/// build.rs from MTVK_CUDA_BATCH_INV_WINDOW); the host's step table must be
/// built for exactly this window.
pub(crate) fn batch_inv_window() -> u32 {
    const WINDOW_STR: &str = env!("MTVK_CUDA_BATCH_INV_WINDOW");
    WINDOW_STR.parse().expect("MTVK_CUDA_BATCH_INV_WINDOW")
}

/// Streaming multiprocessor count of device 0 (CU_DEVICE_ATTRIBUTE_
/// MULTIPROCESSOR_COUNT), used to size the default launch to whole waves:
/// the window-walk kernels fit one 256-thread block per SM, so a batch that
/// is not a multiple of SMs x 256 leaves SMs idle (32768 threads on a 170-SM
/// RTX 5090 is 128 blocks, a quarter of the GPU unused).
pub(crate) fn multiprocessor_count() -> Result<u32> {
    const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: c_int = 16;
    let driver = CudaDriver::load()?;
    driver.initialize()?;
    let count = driver.device_attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?;
    if count <= 0 {
        return Err(anyhow!("CUDA reported {count} multiprocessors"));
    }
    Ok(count as u32)
}

#[cfg(test)]
mod tests {
    use super::CudaSearcher;
    use crate::gpu::Searcher;
    use crate::keygen::{derive_candidate, scalar_words_for_gpu};
    use crate::prefix::Prefix;

    /// Known scalar/pubkey pair: the self-test seed. The public key is
    /// computed by curve25519-dalek's compress(), i.e. the authoritative
    /// canonical 32-byte encoding including the x sign bit.
    fn known_key() -> ([u32; 8], [u8; 32]) {
        let seed = [42u8; 32];
        let candidate = derive_candidate(seed);
        let (words, _) = scalar_words_for_gpu(seed);
        (words, candidate.public_key)
    }

    fn one_key_search(prefix_text: &str, words: &[u32]) -> Option<(usize, u32)> {
        let prefix = Prefix::parse(prefix_text).expect("valid prefix");
        let mut searcher = CudaSearcher::new(1, 1, &prefix).expect("CUDA searcher");
        searcher
            .search_words(words)
            .expect("CUDA search")
            .map(|m| (m.index, m.iter))
    }

    /// The kernel's full 32-byte compressed point (bytes 0..31 including the
    /// sign bit in byte 31) must equal dalek's compress() byte-for-byte.
    #[test]
    fn gpu_debug_point_matches_dalek_compress() {
        let (words, public_key) = known_key();
        // Any valid prefix works; use one that does not match so the search
        // result itself is irrelevant here.
        let prefix = Prefix::parse("ABCD1234").unwrap();
        let mut searcher = CudaSearcher::new(1, 1, &prefix).expect("CUDA searcher");
        let _ = searcher.search_words(&words).expect("CUDA search");
        assert_eq!(searcher.last_debug_point(), public_key, "GPU compressed point != dalek compress()");
    }

    /// Every accepted prefix length of the known key matches at (0, 0) and
    /// a near miss in the last nibble does not.
    #[test]
    fn gpu_matches_known_prefixes_and_rejects_near_misses() {
        let (words, public_key) = known_key();
        let hex_key = hex::encode_upper(public_key);
        for len in 1..=8usize {
            assert_eq!(one_key_search(&hex_key[..len], &words), Some((0, 0)), "length {len}");
            let mut text: Vec<u8> = hex_key[..len].into();
            text[len - 1] = if text[len - 1] == b'0' { b'1' } else { b'0' };
            let text = String::from_utf8(text).unwrap();
            assert_eq!(one_key_search(&text, &words), None, "near miss at length {len}");
        }
    }

    /// The walk must find a key at every window-boundary position: the last
    /// candidate of a full window, the first of the next, the final candidate
    /// of a short tail window, and deep into a multi-window walk.
    #[test]
    fn gpu_search_finds_keys_across_inversion_windows() {
        use crate::keygen::{candidate_from_scalar, scalar_add_u64, scalar_to_words};
        let (_, clamped) = scalar_words_for_gpu([42u8; 32]);
        let words = scalar_to_words(&clamped);
        let w = super::batch_inv_window();
        assert!(w >= 32);
        for (iters, target_iter) in [
            (w + 1, w - 1),
            (w + 1, w),
            (w + 37, w + 36),
            (2 * w + 5, 2 * w + 4),
            (3 * w, 3 * w - 1),
            (w, w - 1),
            (w, 0),
            (w, w / 2),
            (w / 2 + 1, w / 2),
            (w / 2, w / 2 - 1),
            (w / 2, 0),
            (2 * w + 1, 2 * w),
            (3 * w + 2, 3 * w + 1),
        ] {
            let stepped = scalar_add_u64(clamped, 8 * target_iter as u64);
            let pubkey = candidate_from_scalar(stepped, [0u8; 32]).public_key;
            let Ok(prefix) = Prefix::parse(&hex::encode_upper(&pubkey[..4])) else {
                continue; // reserved first byte: not a MeshCore key
            };
            let mut searcher = CudaSearcher::new(1, iters, &prefix).expect("CUDA searcher");
            let found = searcher.search_words(&words).expect("CUDA search");
            assert_eq!(
                found.map(|m| (m.index, m.iter)),
                Some((0, target_iter)),
                "iters={iters} target_iter={target_iter}"
            );
        }
    }
}

/// Meshtastic result block (u32 words): [found_count, found_index[64], found_iter[64], debug_u[8]].
const MESH_RESULT_WORDS: usize = 1 + 64 + 64 + 8;

// ---------------------------------------------------------------------------
// Meshtastic node-ID mode (mesh_kernel): the same launch shape with the
// pattern's parameter block and CRC/bitmap tables and up to 64 hits per
// launch.
// ---------------------------------------------------------------------------

pub(super) struct CudaMeshSearcher {
    params_buffer: DeviceBuffer,
    tables_buffer: DeviceBuffer,
    basepoint_table_buffer: DeviceBuffer,
    step_table_buffer: DeviceBuffer,
    slots: [LaunchSlot; PIPELINE_SLOTS],
    stream: CudaStream,
    submitted: usize,
    collected: usize,
    _module: CudaModule,
    context: CudaContext,
    driver: Arc<CudaDriver>,
    function: CUfunction,
    batch_size: usize,
    iters: u32,
    /// X25519 public key of gid 0's first candidate on the last collected
    /// launch, as eight little-endian words.
    last_debug_u: [u32; 8],
}

impl CudaMeshSearcher {
    pub(super) fn new(
        batch_size: usize,
        iters: u32,
        pattern: &crate::meshtastic::NodeIdPattern,
    ) -> Result<Self> {
        let driver = Arc::new(CudaDriver::load()?);
        unsafe {
            driver.check((driver.cu_init)(0), "cuInit")?;
        }
        let context = CudaContext::create(Arc::clone(&driver))?;
        let module = CudaModule::load(Arc::clone(&driver), context.raw())?;
        let function = module.function("mesh_kernel")?;

        let params = super::MeshParams::new(batch_size, iters, pattern);
        let params_buffer = DeviceBuffer::with_value(Arc::clone(&driver), context.raw(), &params)?;
        let tables_buffer =
            DeviceBuffer::with_slice(Arc::clone(&driver), context.raw(), &pattern.gpu_tables())?;
        let basepoint_table_buffer =
            DeviceBuffer::with_slice(Arc::clone(&driver), context.raw(), table::basepoint_table())?;
        // The d*x*y step table: see mesh_kernel's operand bounds.
        let step_table_buffer = DeviceBuffer::with_slice(
            Arc::clone(&driver),
            context.raw(),
            &ed_step_table::mesh_step_table(batch_inv_window()),
        )?;
        let slots = [
            LaunchSlot::new(&driver, context.raw(), batch_size * 8, MESH_RESULT_WORDS)?,
            LaunchSlot::new(&driver, context.raw(), batch_size * 8, MESH_RESULT_WORDS)?,
        ];
        let stream = CudaStream::new(Arc::clone(&driver), context.raw())?;

        Ok(Self {
            params_buffer,
            tables_buffer,
            basepoint_table_buffer,
            step_table_buffer,
            slots,
            stream,
            submitted: 0,
            collected: 0,
            _module: module,
            context,
            driver,
            function,
            batch_size,
            iters,
            last_debug_u: [0u32; 8],
        })
    }

    /// The X25519 public key the kernel computed for gid 0's first candidate
    /// on the last collected launch.
    #[cfg(test)]
    pub(super) fn last_debug_u(&self) -> [u8; 32] {
        let mut bytes = [0u8; 32];
        for (chunk, word) in bytes.chunks_exact_mut(4).zip(self.last_debug_u) {
            chunk.copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }

    fn enqueue(&mut self, words: &[u32]) -> Result<()> {
        if words.len() != self.batch_size * 8 {
            return Err(anyhow!(
                "expected {} scalar words, got {}",
                self.batch_size * 8,
                words.len()
            ));
        }
        if self.submitted - self.collected >= PIPELINE_SLOTS {
            return Err(anyhow!("CUDA pipeline overfilled: collect a batch before submitting"));
        }
        self.context.set_current()?;
        let slot_index = self.submitted % PIPELINE_SLOTS;
        let block_size = block_size();
        let grid_size = (self.batch_size as u32).div_ceil(block_size);
        let function = self.function;
        let driver = Arc::clone(&self.driver);
        let mut params = self.params_buffer.ptr;
        let mut tables = self.tables_buffer.ptr;
        let mut basepoint_table = self.basepoint_table_buffer.ptr;
        let mut step_table = self.step_table_buffer.ptr;
        let slot = &mut self.slots[slot_index];
        slot.enqueue(words, &self.stream, |scalars_dev, results_dev, stream| {
            let mut scalars = scalars_dev.ptr;
            let mut found_count = results_dev.ptr;
            let mut found_index = results_dev.ptr + 4;
            let mut found_iter = results_dev.ptr + 4 * 65;
            let mut debug_u = results_dev.ptr + 4 * 129;
            let mut kernel_args = [
                &mut scalars as *mut CUdeviceptr as *mut c_void,
                &mut params as *mut CUdeviceptr as *mut c_void,
                &mut tables as *mut CUdeviceptr as *mut c_void,
                &mut found_count as *mut CUdeviceptr as *mut c_void,
                &mut found_index as *mut CUdeviceptr as *mut c_void,
                &mut found_iter as *mut CUdeviceptr as *mut c_void,
                &mut debug_u as *mut CUdeviceptr as *mut c_void,
                &mut basepoint_table as *mut CUdeviceptr as *mut c_void,
                &mut step_table as *mut CUdeviceptr as *mut c_void,
            ];
            driver.launch_on(function, grid_size, block_size, kernel_args.as_mut_ptr(), stream.raw)
        })?;
        self.submitted += 1;
        Ok(())
    }

    fn collect_oldest(&mut self) -> Result<Vec<Match>> {
        if self.collected >= self.submitted {
            return Ok(Vec::new());
        }
        let slot = &mut self.slots[self.collected % PIPELINE_SLOTS];
        slot.wait()?;
        self.collected += 1;
        let results = slot.results.host.as_slice();
        self.last_debug_u.copy_from_slice(&results[129..137]);
        let count = (results[0] as usize).min(64);
        let indexes = &results[1..65];
        let iters_out = &results[65..129];
        let mut hits = Vec::with_capacity(count);
        for i in 0..count {
            let index = indexes[i] as usize;
            let iter = iters_out[i];
            if index >= self.batch_size || iter >= self.iters {
                return Err(anyhow!(
                    "CUDA mesh returned out-of-range match ({index}, {iter})"
                ));
            }
            hits.push(Match { index, iter });
        }
        Ok(hits)
    }
}

impl Drop for CudaMeshSearcher {
    fn drop(&mut self) {
        if self.context.set_current().is_ok() {
            let _ = self.driver.synchronize();
        }
    }
}

impl MeshSearcher for CudaMeshSearcher {
    fn search_words(&mut self, words: &[u32]) -> Result<Vec<Match>> {
        if self.collected != self.submitted {
            return Err(anyhow!("search_words called with a pipelined batch in flight"));
        }
        self.enqueue(words)?;
        self.collect_oldest()
    }

    fn pipeline_depth(&self) -> usize {
        PIPELINE_SLOTS
    }

    fn submit(&mut self, words: &[u32]) -> Result<()> {
        self.enqueue(words)
    }

    fn collect(&mut self) -> Result<Vec<Match>> {
        if self.collected >= self.submitted {
            return Err(anyhow!("collect called with no CUDA batch in flight"));
        }
        self.collect_oldest()
    }
}

#[cfg(test)]
mod mesh_tests {
    use super::CudaMeshSearcher;
    use crate::gpu::MeshSearcher;
    use crate::keygen::{random_base_scalar, scalar_add_u64, scalar_to_words};
    use crate::meshtastic::{crc32_bitwise, derive, oracle_vectors, NodeIdPattern};
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;
    use std::sync::{Mutex, MutexGuard};

    /// Every CUDA context reserves local memory for all resident threads
    /// (20 KB of stack x ~350k threads, ~7 GB on an RTX 5090), so these
    /// tests run one at a time to keep the concurrent test contexts within
    /// the device's memory.
    static SERIAL: Mutex<()> = Mutex::new(());
    fn serial() -> MutexGuard<'static, ()> {
        SERIAL.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn words_of(bases: &[[u8; 32]]) -> Vec<u32> {
        bases.iter().flat_map(|b| scalar_to_words(b)).collect()
    }

    /// The kernel's u of every oracle secret (gid 0's debug word) must equal
    /// the oracle's X25519 public key byte for byte, and with all of a
    /// batch's node ids as the (fully fixed) targets every thread must
    /// report exactly iteration 0.
    #[test]
    fn mesh_kernel_matches_oracle_vectors() {
        let _guard = serial();
        let vectors: Vec<_> = oracle_vectors()
            .into_iter()
            .filter(|(s, _, _)| !s[8..31].iter().all(|b| *b == 0xFF))
            .collect();
        assert!(vectors.len() >= 190);
        for chunk in vectors.chunks(64) {
            let ids: Vec<String> = chunk.iter().map(|(_, _, id)| format!("{id:08x}")).collect();
            let pattern = NodeIdPattern::parse(&ids.join(",")).unwrap();
            let bases: Vec<[u8; 32]> = chunk.iter().map(|(s, _, _)| *s).collect();
            let mut searcher = CudaMeshSearcher::new(bases.len(), 1, &pattern).expect("CUDA mesh searcher");
            let hits = searcher.search_words(&words_of(&bases)).expect("mesh launch");
            let mut got: Vec<(usize, u32)> = hits.iter().map(|h| (h.index, h.iter)).collect();
            got.sort_unstable();
            let wanted: Vec<(usize, u32)> = (0..bases.len()).map(|i| (i, 0)).collect();
            assert_eq!(got, wanted);
            assert_eq!(searcher.last_debug_u(), chunk[0].1, "debug u of {}", hex::encode(chunk[0].0));
        }
        let pattern = NodeIdPattern::parse("0").unwrap();
        let mut searcher = CudaMeshSearcher::new(1, 1, &pattern).expect("CUDA mesh searcher");
        for (secret, pubkey, id) in &vectors {
            let hits = searcher.search_words(&scalar_to_words(secret)).expect("mesh launch");
            assert_eq!(searcher.last_debug_u(), *pubkey, "u of {}", hex::encode(secret));
            assert_eq!(hits.len(), usize::from(pattern.matches(*id)), "hit set of {}", hex::encode(secret));
        }
    }

    /// Keys planted at every window-boundary position of a multi-window
    /// walk (plus random positions) across 64 threads must be reported
    /// exactly, with a fully fixed 8-digit target per plant.
    #[test]
    fn mesh_kernel_reports_exact_planted_set_across_windows() {
        let _guard = serial();
        let w = super::batch_inv_window();
        assert!(w >= 32);
        let iters = 2 * w + 9;
        let batch = 64usize;
        let mut rng = ChaCha20Rng::from_seed([21u8; 32]);
        let bases: Vec<[u8; 32]> = (0..batch).map(|_| random_base_scalar(&mut rng)).collect();
        let mut planted: Vec<(usize, u32)> = vec![
            (0, 0), (1, w / 2 - 1), (2, w / 2), (3, w / 2 + 1), (4, w - 1), (5, w), (6, w + 1),
            (7, 2 * w - 1), (8, 2 * w), (9, 2 * w + 8), (10, iters - 1), (10, 0),
        ];
        while planted.len() < 60 {
            let p = ((rand_core::RngCore::next_u32(&mut rng) as usize) % batch, rand_core::RngCore::next_u32(&mut rng) % iters);
            if !planted.contains(&p) {
                planted.push(p);
            }
        }
        let targets: Vec<String> = planted
            .iter()
            .map(|&(i, iter)| format!("{:08x}", derive(&scalar_add_u64(bases[i], 8 * iter as u64)).1))
            .collect();
        let pattern = NodeIdPattern::parse(&targets.join(",")).unwrap();
        let mut expected: Vec<(usize, u32)> = planted.clone();
        expected.sort_unstable();
        expected.dedup();
        let mut searcher = CudaMeshSearcher::new(batch, iters, &pattern).expect("CUDA mesh searcher");
        for _ in 0..2 {
            let hits = searcher.search_words(&words_of(&bases)).expect("mesh launch");
            let mut got: Vec<(usize, u32)> = hits.iter().map(|h| (h.index, h.iter)).collect();
            got.sort_unstable();
            assert_eq!(got, expected);
        }
    }

    /// Short patterns on random bases: every hit re-derives, no duplicate
    /// reports, and the hit count is consistent with 16^-digits per
    /// candidate (the 64-slot cap for the 1-digit case).
    #[test]
    fn mesh_kernel_hits_rederive_and_rate_is_honest() {
        let _guard = serial();
        let mut rng = ChaCha20Rng::from_seed([22u8; 32]);
        let batch = 256usize;
        let iters = 1024u32;
        let candidates = (batch as u32 * iters) as f64;
        for (text, digits) in [("a", 1u32), ("dc", 2), ("!1*3", 2), ("dead,beef,cafe", 4)] {
            let pattern = NodeIdPattern::parse(text).unwrap();
            let bases: Vec<[u8; 32]> = (0..batch).map(|_| random_base_scalar(&mut rng)).collect();
            let mut searcher = CudaMeshSearcher::new(batch, iters, &pattern).expect("CUDA mesh searcher");
            let hits = searcher.search_words(&words_of(&bases)).expect("mesh launch");
            let mut seen: Vec<(usize, u32)> = hits.iter().map(|h| (h.index, h.iter)).collect();
            seen.sort_unstable();
            seen.dedup();
            assert_eq!(seen.len(), hits.len(), "duplicate reports for {text}");
            for h in &hits {
                let secret = scalar_add_u64(bases[h.index], 8 * h.iter as u64);
                let (pk, id) = derive(&secret);
                assert_eq!(crc32_bitwise(&pk), id);
                assert!(pattern.matches(id), "{text}: hit ({}, {}) -> {id:08x} does not match", h.index, h.iter);
            }
            let expected = candidates * pattern.targets.len() as f64 / 16f64.powi(digits as i32);
            if expected > 64.0 {
                assert_eq!(hits.len(), 64, "{text}: slot cap");
            } else {
                let sigma = expected.sqrt();
                assert!(
                    (hits.len() as f64 - expected).abs() < 5.0 * sigma + 2.0,
                    "{text}: {} hits, expected {expected:.1} +- {sigma:.1}",
                    hits.len()
                );
            }
        }
    }

    /// Pipelined launches return their own hits in submission order.
    #[test]
    fn mesh_pipeline_returns_hits_in_order() {
        let _guard = serial();
        let depth = super::PIPELINE_SLOTS;
        let mut rng = ChaCha20Rng::from_seed([23u8; 32]);
        let base = random_base_scalar(&mut rng);
        let ids: Vec<String> = (0..depth as u64)
            .map(|k| format!("{:08x}", derive(&scalar_add_u64(base, 8 * (16 * k + 3))).1))
            .collect();
        let pattern = NodeIdPattern::parse(&ids.join(",")).unwrap();
        let mut searcher = CudaMeshSearcher::new(1, 16, &pattern).expect("CUDA mesh searcher");
        for k in 0..depth as u64 {
            searcher.submit(&scalar_to_words(&scalar_add_u64(base, 8 * 16 * k))).expect("submit");
        }
        for k in 0..depth {
            let hits = searcher.collect().expect("collect");
            assert_eq!(hits.len(), 1, "launch {k}: {hits:?}");
            assert_eq!((hits[0].index, hits[0].iter), (0, 3), "launch {k}");
        }
        assert!(searcher.collect().is_err());
    }
}
