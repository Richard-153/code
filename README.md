# ReadMe

This repository contains the source code for a multi-GPU diffusion model implementation. This code was finalized in 2023 and is no longer under active maintenance.

## Important Notice

*   **Hardware Requirement:** This project is specifically designed to run on a machine with **three GPUs**. The logic relies on parallel processing across three separate devices and will not work on a single GPU setup.
*   **Environment:** This code was only tested in a single, specific development environment from 2023. You may need to adjust dependencies or configurations to run it successfully on your machine.
*   **Maintenance Status:** The author no longer has access to a three-GPU environment for testing. Consequently, I am unable to provide further validation or technical support for code-related issues. I apologize for any inconvenience this may cause.

## Data Preparation

The methodology requires the dataset to be prepared in a specific way. The core idea is to split image data into three parts based on complexity (measured by standard deviation).

1.  **Split Images:** Divide each training image into three separate parts.
2.  **Calculate Standard Deviation:** For each part, calculate its pixel standard deviation.
3.  **Assign to Models:** The models are designed to handle different levels of complexity:
    *   The part with the **lowest** standard deviation is to be trained by the `diffusion_A` model.
    *   The part with the **middle** standard deviation is to be trained by the `diffusion_B` model.
    *   The part with the **highest** standard deviation is to be trained by the `diffusion_C` model.

## Usage

Please ensure your environment is set up and the data is prepared before proceeding.

### 1. Train the Models
```bash
sh pretreatment.sh
```

### 2. Test the Models
```
sh test.sh
```
## Code Structure
```
.
├── diffusion_A/      # Code for Model A (handles lowest std dev data)
├── diffusion_B/      # Code for Model B (handles middle std dev data)
├── diffusion_C/      # Code for Model C (handles highest std dev data)
├── pretreatment.sh   # Script to start the training process
├── test.sh           # Script to start the evaluation process
└── README.md         # This README file
```

## Contact

Should you have any conceptual questions, feel free to reach out via email. However, please understand that I cannot provide assistance with code debugging.
Email: yebidaxiong2025@163.com

