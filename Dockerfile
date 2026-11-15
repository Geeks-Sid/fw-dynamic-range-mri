#############################################
# Select the OS
FROM python:3.9.7-slim-buster

#############################################
# Setup default flywheel/v0 directory
ENV FLYWHEEL=/flywheel/v0
RUN mkdir -p ${FLYWHEEL}
WORKDIR ${FLYWHEEL}

#############################################
# install required dependencies (before the code, so code changes reuse this layer)
COPY requirements.txt $FLYWHEEL/
RUN pip install -r requirements.txt

#############################################
# copy the gear
COPY run.py manifest.json README.md $FLYWHEEL/
COPY fw_dynamic_range_mri $FLYWHEEL/fw_dynamic_range_mri

#############################################
# Configure entrypoint
ENTRYPOINT ["python", "/flywheel/v0/run.py"]
