# OpenSim Arm26 plant container for NRP_BGA-SB Phase 10.
# Prebuilt OpenSim from the opensim-org conda channel (NOT a from-source build).
FROM condaforge/miniforge3:24.9.2-0

# Install OpenSim (Python bindings + native libs) into the base conda env.
# Pin 4.6; fallback 4.5.x is also on opensim-org if the solve ever fails.
RUN conda install -y -c opensim-org -c conda-forge opensim=4.6 \
    && conda clean -afy

WORKDIR /opt/nrp
COPY models/ /opt/nrp/models/
COPY _arm26_plant.py run_plant.py validate_plant.py /opt/nrp/

# Deterministic integration; headless (no GUI/visualizer).
ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python"]
