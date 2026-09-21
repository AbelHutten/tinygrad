from __future__ import annotations
import subprocess, pathlib, struct, ctypes, tempfile, functools, platform, weakref, threading, array, sys
from tinygrad.helpers import to_mv, round_up, cache_dir, unwrap, prod
import tinygrad.runtime.support.objc as objc
from tinygrad.device import Buffer, BufferStorage, BufferSpec, Allocator, Compiled, Compiler, CompileError, MMIOInterface
from tinygrad.dtype import dtypes, AddrSpace
from tinygrad.renderer.cstyle import MetalRenderer
from tinygrad.runtime.autogen import metal
from tinygrad.runtime.support.c import DLL
from tinygrad.runtime.support.hcq2 import HWQueue, EncodeCtx, encode_submit, ccall, patch, layout_args
from tinygrad.uop.ops import Ops, UOp, UPat, PatternMatcher
from tinygrad.engine.realize import get_call_arg_uops, get_call_var_uops

# 13 is requestType that metal uses to compile source code into MTLB, there aren't any docs or symbols.
REQUEST_TYPE_COMPILE = 13

# Must be loaded for default Metal Device: https://developer.apple.com/documentation/metal/1433401-mtlcreatesystemdefaultdevice?language=objc
DLL("CoreGraphics", "CoreGraphics")

# FIXME: these need autogen to support objc categories
# https://developer.apple.com/library/archive/documentation/Cocoa/Conceptual/ObjectiveC/Chapters/ocCategories.html
@functools.cache
def to_ns_str(s:str): return ctypes.cast(objc.msg("stringWithUTF8String:")(metal.NSString._objc_class_, s.encode()), metal.NSString).own()
def checked(fn, *args): # fn(*args, &error), raised if set
  ret = fn(*args, ctypes.byref(err:=metal.NSError()))
  if err.value is not None: raise RuntimeError(bytes(objc.msg("UTF8String", ctypes.c_char_p)(err.localizedDescription())).decode())
  return ret

class MetalCompiler(Compiler):
  # Opening METAL after LLVM doesn't fail because ctypes.CDLL opens with RTLD_LOCAL but MTLCompiler opens it's own llvm with RTLD_GLOBAL
  # This means that MTLCompiler's llvm will create it's own instances of global state because RTLD_LOCAL doesn't export symbols, but if RTLD_GLOBAL
  # library is loaded first then RTLD_LOCAL library will just use it's symbols. On linux there is RTLD_DEEPBIND to prevent that, but on macos there
  # doesn't seem to be anything we can do.
  import tinygrad.runtime.autogen.llvm as _
  support = DLL("MTLCompiler", "MTLCompiler")
  support.MTLCodeGenServiceCreate.restype = ctypes.c_void_p

  def __init__(self):
    self.cgs = ctypes.c_void_p(MetalCompiler.support.MTLCodeGenServiceCreate(b"tinygrad"))
    super().__init__("compile_metal_direct")
  def __reduce__(self): return (MetalCompiler,()) # force pickle to create new instance for each multiprocessing fork
  def compile(self, src:str) -> bytes:
    ret: Exception|bytes = CompileError("MTLCodeGenServiceBuildRequest returned without calling the callback")
    @ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p)
    def callback(blockptr, error, dataPtr, dataLen, errorMessage):
      nonlocal ret
      if error == 0:
        reply = bytes(to_mv(dataPtr, dataLen))
        # offset from beginning to data = header size + warning size
        ret = reply[sum(struct.unpack('<LL', reply[8:16])):]
      else:
        ret = CompileError(errorMessage.decode())

    # no changes for compute in 2.0 - 2.4 specs, use 2.0 as default for old versions.
    macos_major = int(platform.mac_ver()[0].split('.')[0])
    metal_version = "metal4.0" if macos_major >= 26 else "metal3.1" if macos_major >= 14 else "metal3.0" if macos_major >= 13 else "macos-metal2.0"

    # llvm will create modules.timestamp in cache path and cache compilation of metal stdlib (250ms => 8ms compilation time)
    # note that llvm won't necessarily create anything else here as apple has prebuilt versions of many standard libraries
    params = f'-fno-fast-math -std={metal_version} --driver-mode=metal -x metal -fmodules-cache-path="{cache_dir}" -fno-caret-diagnostics'
    # source blob has to be padded to multiple of 4 but at least one 'b\x00' should be added, params blob just has to be null terminated
    src_padded, params_padded = src.encode() + b'\x00'*(round_up(len(src) + 1, 4) - len(src)), params.encode() + b'\x00'
    request = struct.pack('<QQ', len(src_padded), len(params_padded)) + src_padded + params_padded
    # The callback is actually not a callback but a block which is apple's non-standard extension to add closures to C.
    # See https://clang.llvm.org/docs/Block-ABI-Apple.html#high-level for struct layout.
    # Fields other than invoke are unused in this case so we can just use ctypes.byref with negative offset to invoke field, add blockptr as a first
    # argument and pretend it's a normal callback
    MetalCompiler.support.MTLCodeGenServiceBuildRequest(self.cgs, None, REQUEST_TYPE_COMPILE, request, len(request), ctypes.byref(callback, -0x10))
    if isinstance(ret, Exception): raise ret
    assert ret[:4] == b"MTLB" and ret[-4:] == b"ENDT", f"Invalid Metal library. {ret!r}"
    return ret
  def disassemble(self, lib:bytes):
    with tempfile.NamedTemporaryFile(delete=True) as shader:
      shader.write(lib)
      shader.flush()
      proc = subprocess.Popen(f"cd {pathlib.Path(__file__).parents[2]}/extra/disassemblers/applegpu && python3 compiler_explorer.py {shader.name}",
                              stdout=subprocess.PIPE, shell=True, text=True, bufsize=1)
      for line in unwrap(proc.stdout): print(line, end="")
      ret = proc.wait()
      if ret: print("Disassembler Error: Make sure you have https://github.com/dougallj/applegpu cloned to tinygrad/extra/disassemblers/applegpu")

# *****************
# queue: a linear is an indirect command buffer, a command per kernel binding its args struct. the body sends a few messages per submit

# host words: the device's objc handles, its residency table, then the selectors the body sends
HANDLES = ("queue", "event", "fence", "resources", "count")
SELECTORS = ("commandBuffer", "computeCommandEncoder", "waitForFence:", "updateFence:", "encodeSignalEvent:value:", "endEncoding", "commit",
             "useResources:count:usage:", "executeCommandsInBuffer:withRange:", "concurrentDispatchThreadgroups:threadsPerThreadgroup:",
             "signaledValue")
MSG = {r: metal.dll.bind(r)(metal.dll.objc_msgSend) for r in (None, ctypes.c_void_p)}
def mtl_const(dev, name:str) -> UOp:
  return UOp.placeholder((len(HANDLES) + len(SELECTORS),), dtypes.uint64, 0, device=dev, tag="mtl_consts").index((HANDLES + SELECTORS).index(name))
def mtl_run(dev) -> UOp: return UOp.placeholder((2,), dtypes.uint64, 0, device=dev, volatile=True, tag="mtl_run") # [command buffer, encoder]
def mtl_word(h:UOp, w:UOp) -> UOp: return w.src[0].after(h).index(w.src[1]) # the word w after h

def mtl_call(h:UOp, target:UOp, sel:str, *args:UOp|int, restype=None) -> UOp: # objc_msgSend to the object in the word target, after h
  return ccall(MSG[restype], mtl_word(h, target).load(), mtl_const(h.device, sel).load(),
               *[UOp.const(a, dtypes.uint64) if isinstance(a, int) else a for a in args])

def mtl_msg(h:UOp, target:UOp, sel:str, *args:UOp|int, result:UOp|None=None) -> UOp: # the call chained after h, the returned object kept in result
  call = mtl_call(h, target, sel, *args, restype=ctypes.c_void_p if result is not None else None)
  while h.op is Ops.AFTER: h = h.src[0] # the chain hangs off the call alone: the symbolic pass unrolls an after of afters link by link
  return h.after(call if result is None else mtl_word(h, result).store(call))

def mtl_poll(tl:UOp) -> UOp: # the timeline is the device's shared event: the fence of a batch polls it in place of the timeline word
  h = mtl_run(tl.device).after(*tl.src[1:]) if tl.op is Ops.AFTER else mtl_run(tl.device)
  return mtl_call(h, mtl_const(tl.device, "event"), "signaledValue", restype=ctypes.c_void_p)

def mtl_stack(*vals:UOp|int) -> UOp: # a stack array of uint64: the by-reference arguments of a call
  r = UOp.placeholder((len(vals),), dtypes.uint64, addrspace=AddrSpace.REG)
  return r.after(*[r.index(i).store(v.cast(dtypes.uint64) if isinstance(v, UOp) else UOp.const(v, dtypes.uint64)) for i, v in enumerate(vals)])

class MetalQueue(HWQueue):
  dev:MetalDevice
  def __init__(self, ctx:EncodeCtx, submit:UOp):
    super().__init__(ctx, submit)
    self.rows, self.cmds, self.dyn, self.stamps, self.end = list[tuple[int, UOp]](), list[tuple](), list[tuple[int, tuple]](), list[UOp](), 0
    self.q(UOp.const(0, dtypes.uint64)) # the generic cmdbuf holds nothing, but must exist

  def exec(self, call:UOp, prg:UOp):
    bufs, vals, obj = get_call_arg_uops(call), get_call_var_uops(call, prg), prg.to_elf()
    args = [bufs[i].getaddr(self.devs) for i in prg.arg.globals] + [v.ccast(var.dtype) for v, var in zip(vals, prg.arg.vars)]
    self.rows += (rows:=layout_args(args, off:=round_up(self.end, 256)))
    self.end = max([o + w.dtype.itemsize for o, w in rows], default=off + 8)
    dims = (*prg.arg.global_size, *prg.arg.local_size)
    if any(isinstance(d, UOp) for d in dims): self.dyn.append((len(self.cmds), dims))
    self.cmds.append((obj.lib, obj.name, tuple(1 if isinstance(d, UOp) else int(d) for d in dims), off))

  def wait(self, dst:UOp, val:UOp, eq=False): pass # one queue: the fence orders its command buffers
  def timestamp(self, dst:UOp): self.stamps.append(dst)
  def signal(self, dst:UOp, val:UOp): self.value = val # the one signal of a single queue: the timeline, the event

  def submit(self, cmdbuf:UOp) -> UOp:
    # the args buffer, made with the icb at link: the kernels' structs, then [icb, the commands...]. the batch's fence waited for the previous run
    n, size = len(self.cmds), round_up(self.end, 8)
    desc = UOp.placeholder((size + 8 * (1 + n),), dtypes.uint8, device=self.devs, volatile=True, tag=("mtl_icb", tuple(self.cmds), size))
    words = (h:=patch(desc, self.rows)).bitcast(dtypes.uint64)[size // 8:]
    cb, enc = mtl_run(self.devs).index(0), mtl_run(self.devs).index(1)
    h = mtl_msg(h, mtl_const(self.devs, "queue"), "commandBuffer", result=cb)
    h = mtl_msg(h, cb, "computeCommandEncoder", result=enc)
    h = mtl_msg(h, enc, "waitForFence:", mtl_const(self.devs, "fence").load()) # metal doesn't track what a kernel reaches: the fence orders the runs
    if self.dev.residency is None: # and the run declares the device's buffers (paravirtual metal has no residency sets)
      h = mtl_msg(h, enc, "useResources:count:usage:", mtl_const(self.devs, "resources").load(), mtl_const(self.devs, "count").load(), 3)
    for ci, dims in self.dyn: # arm64 passes MTLSize by reference
      h = mtl_msg(h, words.index(1 + ci), "concurrentDispatchThreadgroups:threadsPerThreadgroup:", (sz:=mtl_stack(*dims)).index(0), sz.index(3))
    h = mtl_msg(h, enc, "executeCommandsInBuffer:withRange:", mtl_word(h, words.index(0)).load(), 0, n)
    h = mtl_msg(h, enc, "updateFence:", mtl_const(self.devs, "fence").load())
    h = mtl_msg(h, enc, "endEncoding")
    h = mtl_msg(h, cb, "encodeSignalEvent:value:", mtl_const(self.devs, "event").load(), self.value)
    if self.stamps: # the command buffer is the batch's stamp until synchronize resolves it, a zero end marks it pending
      h = h.after(mtl_word(h, self.stamps[0].index(1)).store(mtl_word(h, cb).load()), mtl_word(h, self.stamps[-1].index(1)).store(0))
    return mtl_msg(h, cb, "commit")

# *****************
# device

class MetalAllocator(Allocator['MetalDevice']):
  def _alloc(self, size:int, options:BufferSpec) -> BufferStorage:
    mtl = metal.MTLBuffer(options.external_ptr) if options.external_ptr else \
          self.dev.sysdevice.newBufferWithLength_options(size, metal.MTLResourceStorageModeShared)
    if mtl.value is None: raise MemoryError(f"Metal OOM while allocating {size=}")
    self.dev.resident(mtl, True)
    return BufferStorage(mtl.gpuAddress(), mtl, MMIOInterface(c, size) if (c:=mtl.contents()) else None) # an external buffer may have no host side
  def do_free(self, storage:BufferStorage, options:BufferSpec): # nothing tracks what the kernels reach: the gpu must be done with a buffer first
    self.dev.synchronize()
    self.dev.resident(storage.meta, False)
    super().do_free(storage, options)
  def _free(self, storage:BufferStorage, options:BufferSpec): # released now, not when the storage is collected
    storage.meta.retain = False
    storage.meta.release()
  def _offset(self, buf:int, size:int, offset:int) -> int: return buf + offset

class MetalDevice(Compiled):
  has_copy_queue = False
  pm_encode = PatternMatcher([
    (UPat(Ops.CUSTOM_FUNCTION, arg="submit_metal_compute", name="submit"), lambda ctx, submit: encode_submit(MetalQueue(ctx, submit))),
  ])
  pm_lower = PatternMatcher([
    (UPat(Ops.LOAD, src=(UPat(Ops.INDEX, src=(UPat(Ops.PARAM, tag="timeline").or_after(name="tl"), UPat(Ops.CONST, arg=0))),)), mtl_poll),
  ])

  def __init__(self, device:str=""):
    self.sysdevice = metal.MTLCreateSystemDefaultDevice()
    self.queue = self.sysdevice.newCommandQueueWithMaxCommandBufferCount(1024)
    if self.queue.value is None: raise RuntimeError("Cannot allocate a new command queue")
    self.event, self.fence = self.sysdevice.newSharedEvent(), self.sysdevice.newFence() # the timeline, and the order of the command buffers
    # what the kernels reach must be resident: everything the device allocates is, in a residency set or (paravirtual metal has none) a table
    rsd = ctypes.cast(objc.msg("new", clsmeth=True)(metal.MTLResidencySetDescriptor), metal.MTLResidencySetDescriptor)
    self.residency = self.sysdevice.newResidencySetWithDescriptor_error(rsd, None)
    if self.residency.value is None: self.residency = None
    else: self.queue.addResidencySet(self.residency)
    self.resources, self.table = list[int](), (ctypes.c_uint64 * 1)()
    self.icbs, self.profile_slots = weakref.WeakKeyDictionary[Buffer, tuple](), weakref.WeakSet[Buffer]() # live with their buffers

    # https://developer.apple.com/documentation/metal/mtlgpufamily
    def check_family(f): return next(filter(self.sysdevice.supportsFamily, reversed([v for v, nm in metal.enum_MTLGPUFamily.items() if f in nm])), 0)
    super().__init__(device, MetalAllocator(self), [MetalRenderer], None,
                     arch=metal.enum_MTLGPUFamily[check_family("Apple") or check_family("Mac")][12:])
    self.pm_bufferize = PatternMatcher([
      (UPat(Ops.PARAM, tag="mtl_consts"), lambda ctx: ctx.consts),
      (UPat(Ops.PARAM, tag="slots", name="b"), lambda ctx, b: ctx.new_slots(b.max_numel()) if b.max_numel() > 4 else None), # stamps, not signals
      (UPat(Ops.PARAM, name="b"), lambda ctx, b: ctx.new_icb(*b.tag[1:]) if isinstance(b.tag, tuple) and b.tag[0] == "mtl_icb" else None),
    ]) + self.pm_bufferize

  def resident(self, mtl:metal.MTLBuffer, add:bool):
    if self.residency is not None:
      objc.msg("addAllocation:" if add else "removeAllocation:", None, [objc.id_])(self.residency, mtl)
      return objc.msg("commit", None)(self.residency)
    self.resources.append(unwrap(mtl.value)) if add else self.resources.remove(unwrap(mtl.value))
    self.table = (ctypes.c_uint64 * max(len(self.resources), 1))(*self.resources)
    if "consts" in self.__dict__: self.consts.host.view(fmt='Q')[3:5] = array.array('Q', [ctypes.addressof(self.table), len(self.resources)])

  @functools.cached_property
  def consts(self) -> Buffer: # the handles, then the selectors: words, not consts of the body, which is cached across processes
    vals = [self.queue.value, self.event.value, self.fence.value, ctypes.addressof(self.table), len(self.resources),
            *[objc.getsel(s.encode()).value for s in SELECTORS]]
    return Buffer(self.host, len(vals), dtypes.uint64, initial_value=struct.pack(f"{len(vals)}Q", *vals))

  def new_slots(self, n:int) -> Buffer: # a profiled batch's stamps, filled at synchronize
    self.profile_slots.add(buf:=Buffer(self.host, n, dtypes.uint64, initial_value=bytes(8 * n)))
    return buf

  @functools.cache
  def pipeline(self, lib:bytes, name:str) -> metal.MTLComputePipelineState:
    library = checked(self.sysdevice.newLibraryWithData_error, objc.dispatch_data_create(lib, len(lib), None, None))
    descriptor = metal.MTLComputePipelineDescriptor.new()
    descriptor.setComputeFunction(library.newFunctionWithName(to_ns_str(name)))
    descriptor.setSupportIndirectCommandBuffers(True)
    return checked(self.sysdevice.newComputePipelineStateWithDescriptor_options_reflection_error, descriptor, metal.MTLPipelineOptionNone, None)

  def new_icb(self, cmds:tuple[tuple[bytes, str, tuple[int, ...], int], ...], size:int) -> Buffer: # a linear's args buffer and its icb
    buf = Buffer(self.device, size + 8 * (1 + len(cmds)), dtypes.uint8, options=BufferSpec(nolru=True), preallocate=True)
    desc = metal.MTLIndirectCommandBufferDescriptor.new()
    desc.setCommandTypes(metal.MTLIndirectCommandTypeConcurrentDispatch)
    desc.setMaxKernelBufferBindCount(1)
    icb = self.sysdevice.newIndirectCommandBufferWithDescriptor_maxCommandCount_options(desc, max(len(cmds), 1), 0)
    if icb.value is None: raise RuntimeError("create indirect command buffer failed, does your system support this?")
    objs = [icb.indirectComputeCommandAtIndex(i).own() for i in range(len(cmds))]
    for cmd, (lib, name, dims, off) in zip(objs, cmds):
      cmd.setComputePipelineState(state:=self.pipeline(lib, name))
      if prod(dims[3:]) > (mx:=state.maxTotalThreadsPerThreadgroup()): raise RuntimeError(f"local size {dims[3:]} bigger than {mx}")
      cmd.setKernelBuffer_offset_atIndex(buf.get_storage().meta, off, 0)
      cmd.concurrentDispatchThreadgroups_threadsPerThreadgroup(metal.MTLSize(*dims[:3]), metal.MTLSize(*dims[3:]))
      cmd.setBarrier() # the kernels run in order
    buf.host.view(fmt='Q')[size // 8:] = array.array('Q', [icb.value, *[c.value for c in objs]])
    self.icbs[buf] = (icb, objs)
    return buf

  def _wait_signal(self, sig:MMIOInterface|memoryview, value:int, timeout:int|None=None): # the timeline is the event
    if sys.is_finalizing(): return # it no longer wakes at interpreter exit, and nothing is left to wait for
    wait = objc.msg("waitUntilSignaledValue:timeoutMS:", ctypes.c_bool, [ctypes.c_uint64, ctypes.c_uint64])
    if not wait(self.event, value, int(self.wait_timeout_ms)): raise RuntimeError(f"{self.device} signal wait timed out") # can't recover: no timeout

  def synchronize(self, timeout:int|None=None):
    for buf in list(self.profile_slots): # a batch's command buffer spans its kernels: they share its time evenly (no per kernel timestamps)
      if (words:=buf.host.view(fmt='Q'))[5] and not words[-1]:
        (cb:=metal.MTLCommandBuffer(words[5])).waitUntilCompleted()
        st, en, n = cb.GPUStartTime() * 1e9, cb.GPUEndTime() * 1e9, (buf.size - 4) // 4
        for k in range(2 * n): words[5 + 2 * k] = int(st + (en - st) * (k // 2 + k % 2) / n)
    super().synchronize(timeout)
    if threading.current_thread() is threading.main_thread(): # release the command buffers: they are autoreleased, the pool is drained here
      objc.lib.objc_autoreleasePoolPop(pool.pop())
      pool.append(objc.lib.objc_autoreleasePoolPush())

pool = [objc.lib.objc_autoreleasePoolPush()]
