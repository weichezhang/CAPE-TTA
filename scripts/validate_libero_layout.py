from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import libero.libero as libero_pkg

print("LIBERO package:", libero_pkg.__file__)
print("Benchmark module:", benchmark.__file__)
print("BDDL path:", get_libero_path("bddl_files"))
print("Init path:", get_libero_path("init_states"))
print("OffScreenRenderEnv:", OffScreenRenderEnv)
