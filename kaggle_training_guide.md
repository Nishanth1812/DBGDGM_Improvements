# Kaggle Training Guide: MM-DBGDGM

This guide outlines the step-by-step process to train the MM-DBGDGM model on Kaggle using Dual T4 GPUs.

## Phase 1: Prepare Local Files

You need to create two ZIP archives from your local machine to upload to Kaggle.

1. **Zip the Data:**
   - Navigate to `C:\WeKan Training Data\`.
   - Compress the `mm_dbgdgm_prepared` folder into a zip file (e.g., `mm_dbgdgm_prepared.zip`).
   *(Note: The `labels.csv` inside has already been updated to use Kaggle-compatible relative paths).*

2. **Zip the Code:**
   - Navigate to your project directory: `H:\Personal\Internships\WeKan\DBGDGM_Improvements\`.
   - Compress the `MM_DBGDGM` folder into a zip file (e.g., `mm_dbgdgm_code.zip`).

## Phase 2: Upload to Kaggle

1. Go to [Kaggle Datasets](https://www.kaggle.com/datasets) and click **New Dataset**.
2. Upload `mm_dbgdgm_prepared.zip` and name the dataset `mm-dbgdgm-prepared`. Click **Create**.
3. Create a second **New Dataset**, upload `mm_dbgdgm_code.zip`, and name it `mm-dbgdgm-code`. Click **Create**.

## Phase 3: Configure the Kaggle Notebook

1. Go to [Kaggle Code](https://www.kaggle.com/code) and click **New Notebook**.
2. **Import the Notebook**:
   - Go to **File > Import Notebook**.
   - Select your local `full_pipeline_demo_updated.ipynb` file.
3. **Add the Datasets**:
   - On the right sidebar, click **Add Input** (or **Add Data**).
   - Search for your datasets: `mm-dbgdgm-prepared` and `mm-dbgdgm-code`.
   - Click the `+` icon next to both to mount them to your notebook.
4. **Enable GPUs**:
   - On the right sidebar, look for **Session options** or **Notebook options**.
   - Change the **Accelerator** to **GPU T4 x2**.
   - Turn on the internet switch if it isn't already.

## Phase 4: Run Training

1. Once the session starts and the GPU is allocated, simply click **Run All**.
2. The notebook has built-in logic to:
   - Detect that it is running on Kaggle.
   - Automatically mount the `MM_DBGDGM` code directory to the system path.
   - Automatically locate the `mm-dbgdgm-prepared` dataset.
   - Detect the Dual T4 GPUs and start the 5-fold cross-validation pipeline.

> **Note on Multi-GPU**: The underlying model relies heavily on dynamic PyTorch Geometric graphs. If you see warnings or errors regarding `DataParallel` struggling to chunk the graph lists across two GPUs, you can safely drop the Accelerator down to a single **GPU T4** in the right sidebar. A single T4 has 16GB of VRAM and easily handles this batch size!
