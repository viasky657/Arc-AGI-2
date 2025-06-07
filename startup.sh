#!/bin/bash
pip install -r requirements.txt
#jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' #This is off in favor of the custom file path version below. 
jupyter lab --ip=0.0.0.0 \
            --port=8888 \
            --no-browser \
            --allow-root \
            --NotebookApp.token='' \
            --notebook-dir=/workspaces/Arc-AGI-2/contineous-thought-machines/examples/Arc_AGI_2_Final.ipynb