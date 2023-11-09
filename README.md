# Our Housing-Design README
This project presents a comprehensive solution to the housing design challenge, rooted in collaboration between architects and advanced technology. Through the use of Heterogeneous Layout Graphs and an effective pipeline, we aim to transform the housing design process, making it more efficient, accessible, and aligned with user requirements while preserving the critical role of architects in the process. We augment the Deep Generative Model of Graphs (DGMG*) from [Li et al., 2018] to create heterogeneous graphs. Hetero-HouseGAN++ from [Nauata et al., 2021] is implemented for the final generation of architecturalquality floor plans.
## Steps to install our project:
### 1. Clone repository:
```
cd ~
git clone https://github.com/jlhofland/housing-design.git
```
### 2. Install conda environment
```
cd housing-design/
conda env create -f environment.yml
y
conda activate house
```
### 3. Install our python package
```
cd housingpipeline
pip install .
```

## Steps to run housing design pipeline
```
cd ~/housing-design/housingpipeline/housingpipeline/floor_plan_pipeline
python main.py
```
