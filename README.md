# minitensor

A systems-first tensor runtime in Rust. Built to explore the layers between a high-level
tensor API and the metal — IR design, execution backends, memory layout, and eventually
JIT compilation.

This is a learning + research project, not a PyTorch replacement. The goal is depth in
runtime and compiler internals, not breadth of ops.

## Why

Most ML frameworks hide the interesting parts: how shapes propagate through an IR, how
kernels get dispatched, how memory is laid out and reused, how a compute graph becomes
machine code. minitensor is an attempt to build those layers from scratch and understand
them by writing them.

## Architecture

<img width="785" height="350" alt="image" src="https://github.com/user-attachments/assets/c99a61e9-766c-470a-ab46-78c4d3f6a4ed" />


Three layers, kept deliberately separate:

- **Storage** — `Arc<[f32]>` with copy-on-write semantics. Cheap clones, explicit ownership
  transitions, no hidden allocations.
- **Tensor** — shape, strides, offset over a storage buffer. Views and reshapes are
  metadata-only; no data copies until forced.
- **IR + execution** — shape-aware intermediate representation, lowered to an execution
  backend. Currently planning a bytecode VM as the first backend, with a JIT path after.

## Status

- [x] Storage layer with Arc-backed CoW
- [x] Tensor representation (shape, strides, offset)
- [ ] Shape-aware IR (in progress)
- [ ] Bytecode VM execution backend
- [ ] JIT backend (Cranelift or hand-rolled)
- [ ] Benchmarks vs Candle / Burn on small workloads

## Roadmap

Near-term goal: end-to-end execution of a small model (e.g., linear regression on tabular
data) through the full pipeline — API → IR → VM — with measurable performance numbers.

Longer-term: JIT compilation, fused kernels, and exploring what a tensor DSL with proper
IR transforms looks like in Rust.

## Non-goals

- Feature parity with PyTorch / Candle
- GPU support (for now — CUDA is a separate learning track)
- Production use

## Author

Built by [Suryakant Soni](https://github.com/Suryakant-Soni). Backend engineer working
toward low-level systems and ML infra.
