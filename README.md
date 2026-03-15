# 🧠 MicroSholl — Advanced Microglia Morphology Python-based Analysis Pipeline

## Introduction
Microglia are the resident immune cells of the central nervous system and play crucial roles in brain development and function. Understanding their morphology is key to studying their roles in health and disease. The Sholl analysis is a widely-used method to quantify the complexity of microglial morphologies by measuring the intersection of dendritic processes with concentric circles around a central point.

## Comparison: MicroSholl vs ImageJ
| Feature                  | MicroSholl               | ImageJ                 |
|--------------------------|--------------------------|------------------------|
| User interface           | Modern GUI               | Basic GUI              |
| Patch processing         | Yes                      | No                     |
| Automated denoising      | Yes                      | Limited options        |
| Batch processing         | Yes                      | Limited                |
| Output customization     | Yes                      | Standard               |

## What's New in v2.0
- Enhanced user interface
- Improved algorithms for preprocessing and denoising
- New batch processing capabilities
- Interactive Quality Control (QC) dashboard

## Pipeline Overview
```
+-----------------------+
|  MicroSholl Pipeline  |
+-----------------------+
|    Preprocessing     |
|      Denoising      |
|    Soma Selection    |
|   Sholl Analysis     |
| Interactive QC Dash.  |
|    Batch Processing   |
| Stat. Post-processing |
+-----------------------+
```  

## Feature Deep-Dive
### A. Patch-wise Adaptive Preprocessing
| Images                              | Description                       |
|-------------------------------------|-----------------------------------|
| ![M_02_global.png](M_02_global.png) | Global adaptive preprocessing image |
| ![M_02_patchwise.png](M_02_patchwise.png) | Patch-wise preprocessing image     |

### B. Automated Denoising
![h_value_UI.png](h_value_UI.png)
Scientific Rationale: Automated denoising is critical in ensuring accurate analysis by removing noise without compromising true morphological features.

### C. Soma Selection
![soma_sel_UI.png](soma_sel_UI.png)
The soma selection interface allows for precise identification and isolation of the soma in microglia.

### D. Sholl Analysis
![sholl_UI.png](sholl_UI.png)
| Metrics      | Description                      |
|--------------|----------------------------------|
| Number of intersections | Total intersections with concentric circles |
| Maximum branching       | Maximum distance from the soma  |
| Complexity index        | Ratio of total length to number of intersections |

### E. Interactive QC Dashboard
![qc_dashboard_example.png](qc_dashboard_example.png)
Panel Descriptions:
- Overview of processing steps
- Quality metrics visualization
- Option to adjust parameters and reprocess

### F. Batch Processing
![batch_dialog.png](batch_dialog.png)
CSV Example:
| Sample ID | Condition | Result | 
|-----------|-----------|--------| 
| Sample_1 | Control   | Pass   | 
| Sample_2 | Treatment  | Fail   | 

### G. Statistical Post-processing
![Sham CB.png](Sham%20CB.png)

## Output Data Format
The output data is structured as follows:
- CSV files with quantitative results
- Visualization images for review

## Repository Structure
- `/src` - Source code
- `/docs` - Documentation
- `/examples` - Example data and scripts

## Installation Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/LorenzoZam/microglia-sholl-pipeline_v2.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Using the command line, you can run:
```bash
python run_microsholl.py --input <input_file> --output <output_file>
```

## Replicability Rationale
| Aspect        | Reasoning                      |
|---------------|--------------------------------|
| Code          | Open source                    |
| Data          | Utilizes standard public datasets |
| Methods       | Well-established algorithms     |

## Technologies Used
| Dependency            | Version          |
|-----------------------|------------------|
| NumPy                 | 1.21.0           |
| SciPy                 | 1.7.0            |
| Matplotlib            | 3.4.0            |
| Other Libraries       | ...              |

## Example Image Attribution
Image sourced from [BioImage Archive S-BIAD1280](https://bioimage.org/IBID/12345).

## Citation
Zammariello L. (2025). Advanced Microglia Morphology Python-based Analysis Pipeline. BioImage Archive.