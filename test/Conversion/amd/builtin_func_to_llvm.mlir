// RUN: triton-opt %s --convert-triton-amdgpu-to-llvm="gfx-arch=gfx942 ftz=True" --convert-builtin-func-to-llvm="ftz=True" | FileCheck %s --check-prefix=LLVM_FTZ
// RUN: triton-opt %s --convert-triton-amdgpu-to-llvm="gfx-arch=gfx950 ftz=True" --convert-builtin-func-to-llvm="ftz=True" | FileCheck %s --check-prefix=LLVM_FTZ
// RUN: triton-opt %s --convert-triton-amdgpu-to-llvm="gfx-arch=gfx942 ftz=False" --convert-builtin-func-to-llvm="ftz=False" | FileCheck %s --check-prefix=LLVM_NO_FTZ
// RUN: triton-opt %s --convert-triton-amdgpu-to-llvm="gfx-arch=gfx950 ftz=False" --convert-builtin-func-to-llvm="ftz=False" | FileCheck %s --check-prefix=LLVM_NO_FTZ
// RUN: triton-opt %s --convert-triton-amdgpu-to-llvm="gfx-arch=gfx1250 ftz=True" --convert-builtin-func-to-llvm="gfx-arch=gfx1250 ftz=True" | FileCheck %s --check-prefix=GFX1250
// RUN: triton-opt %s --convert-triton-amdgpu-to-llvm="gfx-arch=gfx942 ftz=True" --convert-builtin-func-to-llvm="gfx-arch=gfx942 ftz=True" | FileCheck %s --check-prefix=GFX942

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @test_fast_expf(%arg0: tensor<64xf32, #blocked>) -> tensor<64xf32, #blocked> {
    // CHECK-LABEL: test_fast_expf
    // LLVM_FTZ: rocdl.exp2
    // LLVM_NO_FTZ: llvm.intr.exp2
    %0 = tt.extern_elementwise %arg0 {libname = "libdevice", libpath = "", pure = true, symbol = "__triton_hip_fast_expf"} : (tensor<64xf32, #blocked>) -> tensor<64xf32, #blocked>
    tt.return %0 : tensor<64xf32, #blocked>
  }
}

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @test_fast_tanhf(%arg0: tensor<64xf32, #blocked>) -> tensor<64xf32, #blocked> {
    // CHECK-LABEL: test_fast_tanhf
    // LLVM_FTZ: rocdl.exp2
    // LLVM_NO_FTZ: llvm.intr.exp2
    %0 = tt.extern_elementwise %arg0 {libname = "libdevice", libpath = "", pure = true, symbol = "__triton_hip_fast_tanhf"} : (tensor<64xf32, #blocked>) -> tensor<64xf32, #blocked>
    tt.return %0 : tensor<64xf32, #blocked>
  }
}

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @test_tanh_bf16(%arg0: tensor<64xbf16, #blocked>) -> tensor<64xbf16, #blocked> {
    // GFX1250-LABEL: test_tanh_bf16
    // GFX1250-NOT: llvm.call @__triton_hip_tanh_bf16
    // GFX1250: llvm.call_intrinsic "llvm.amdgcn.tanh.bf16"(%{{.*}}) : (bf16) -> bf16
    // GFX1250-NOT: llvm.call @__triton_hip_tanh_bf16
    // GFX942-LABEL: test_tanh_bf16
    // GFX942-NOT: llvm.amdgcn.tanh
    // GFX942: %[[EXT:.*]] = llvm.fpext %{{.*}} : bf16 to f32
    // GFX942: %[[TANH:.*]] = llvm.call @__ocml_tanh_f32(%[[EXT]]) : (f32) -> f32
    // GFX942: llvm.fptrunc %[[TANH]] : f32 to bf16
    // GFX942-NOT: llvm.call @__triton_hip_tanh_bf16
    %0 = tt.extern_elementwise %arg0 {libname = "", libpath = "", pure = true, symbol = "__triton_hip_tanh_bf16"} : (tensor<64xbf16, #blocked>) -> tensor<64xbf16, #blocked>
    tt.return %0 : tensor<64xbf16, #blocked>
  }
}
