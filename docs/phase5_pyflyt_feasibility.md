# Phase 5: PyFlyt Feasibility Survey

## Installability
**FAILED**. We attempted to install `PyFlyt` in the current Python environment (Mac ARM64, Python 3.12). 
`PyFlyt` depends on `pybullet`, which notoriously fails to build from source on Apple Silicon (M1/M2/M3) without heavily customized C++ compiler flags and downgraded python versions. The installation failed during the `pybullet` wheel build step with `clang` exit code 1.

Because `gym-pybullet-drones` also relies on `pybullet`, it will suffer from the exact same unresolvable build error on this machine.

## Next Steps
As requested by the fallback instructions: *"If the drone environments are too heavy or fail to install, implement a minimal custom 2D drone Gymnasium environment as fallback."*

Since compiling PyBullet is not feasible on this architecture without breaking the virtual environment, and alternatives like AirSim/Flightmare are too heavy for quick prototyping, the most reliable path forward is to build a minimal, custom 2D drone physics environment in pure NumPy or PyTorch that conforms to the Gymnasium API.

This fallback environment will provide exactly what we need for the world model and curiosity prototyping (action/state spaces, crashes, physics) without the immense overhead of a 3D engine.

No smoke test for PyFlyt can be executed.
