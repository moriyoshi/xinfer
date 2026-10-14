use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=src/models/nemotron_h/expert_gemm/quant_gather.cu");
    if env::var_os("CARGO_FEATURE_NEMOTRON_EXPERT_FUSED_QUANT").is_some() {
        let out = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));
        let object = out.join("nemotron_expert_quant_gather.o");
        let library = out.join("libxinfer_nemotron_expert_quant.a");
        let nvcc = env::var_os("CUDACXX").unwrap_or_else(|| "nvcc".into());
        let status = Command::new(nvcc)
            .arg("-std=c++17")
            .arg("-O3")
            .arg("-arch=sm_121f")
            .arg("-Xcompiler=-fPIC")
            .arg("-c")
            .arg("src/models/nemotron_h/expert_gemm/quant_gather.cu")
            .arg("-o")
            .arg(&object)
            .status()
            .expect("launch nvcc for Nemotron expert quantization");
        assert!(status.success(), "compile Nemotron expert quantization");
        let status = Command::new("ar")
            .arg("crus")
            .arg(&library)
            .arg(&object)
            .status()
            .expect("launch ar for Nemotron expert quantization");
        assert!(status.success(), "archive Nemotron expert quantization");
        println!("cargo:rustc-link-search=native={}", out.display());
        println!("cargo:rustc-link-lib=static=xinfer_nemotron_expert_quant");
        println!("cargo:rustc-link-lib=dylib=cudart");
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
    let source_dir = PathBuf::from("src/models/nemotron_h/attention_gemm");
    for name in [
        "launch.cu",
        "flash_prefill_paged.cuh",
        "flash_sm_compat.cuh",
    ] {
        println!("cargo:rerun-if-changed={}", source_dir.join(name).display());
    }
    if env::var_os("CARGO_FEATURE_NEMOTRON_ATTENTION_GEMM").is_none() {
        return;
    }
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));
    let object = out.join("nemotron_flash_prefill.o");
    let library = out.join("libxinfer_nemotron_flash.a");
    let nvcc = env::var_os("CUDACXX").unwrap_or_else(|| "nvcc".into());
    let status = Command::new(nvcc)
        .arg("-std=c++17")
        .arg("-O3")
        .arg("-arch=sm_120")
        .arg("-Xcompiler=-fPIC")
        .arg("-c")
        .arg(source_dir.join("launch.cu"))
        .arg("-o")
        .arg(&object)
        .status()
        .expect("launch nvcc");
    assert!(status.success(), "compile Nemotron flash prefill");
    let status = Command::new("ar")
        .arg("crus")
        .arg(&library)
        .arg(&object)
        .status()
        .expect("launch ar");
    assert!(status.success(), "archive Nemotron flash prefill");
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=xinfer_nemotron_flash");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
