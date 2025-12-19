Milestone 1 (“Runnable MVP”) is now in place — the GPUMCRPTDosimetry package can run end-to-end and produce an energy deposition tensor, with all previously-broken imports resolved.

What I changed

Fixed a hard blocker: load_physics_tables_h5() now returns tables in tables.py.
Fixed pipeline output: run_dosimetry() now writes uncertainty as well in pipeline.py.
Added the missing subpackages that current code/tests import:
Materials: hu_materials.py
NIfTI I/O: nifti.py
Dose scoring: scoring.py
CPU oracle stub: cpu_oracle.py
Added an MVP transport engine that makes the system runnable immediately:
engine_gpu_triton_localdepositonly.py
Wired TransportEngine to use it in engine.py
Note: this MVP engine deposits all particle energy locally (not full transport yet). It’s explicitly a Milestone-1 unblocker.
Made toy table builder robust when --out is just a filename: build_toy_physics_h5.py
Added a runnable end-to-end script: run_toy_mvp.py
Expanded config into a pipeline-compatible template: example_simulation.yaml
Added pytest to deps: requirements.txt
How to run the MVP now

Build toy physics tables: python scripts/build_toy_physics_h5.py --out toy_physics.h5
Run the MVP end-to-end: PYTHONPATH=src python scripts/run_toy_mvp.py --device cpu --tables toy_physics.h5 --histories 10000 --out toy_edep.pt
Next development (Milestone 2 suggestion)

Replace the MVP local-deposition engine with the first real wavefront step (even photon-only: flight + Compton/PE + energy bookkeeping gates). If you want, I can start by making a “photon-only transport” engine that still keeps the same TransportEngine interface.