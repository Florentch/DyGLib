# README - Internship from 02/06/2025 to 22/08/2025

## Work Presentation

Presentation of the work carried out during this internship available here:
https://www.canva.com/design/DAGvGUVs_ys/RRTtlwAVBJpH8CiiuXAAnQ/edit?ui=e30

## How to Reproduce the Results and Graphics

### Optc data preprocessing

Optc data format : .pikle
dyglib required format : .csv with [source node, destination node, timestamp (double), label, features 1, features 2,...]  

→ Execute the `optc_data_preprocessor.py` file to obtain the CSV files for machines 051 and 201 in the `DG_data` folder

python optc_data_preprocessor.py --input_path path/to/pkl/data/ --output_path DG_data/data/name/.csv --label_path /path/to/label/.json

→ Perform preprocessing of the created CSV files by running in 'preprocess_data' folder:
```bash
python preprocess_data.py --dataset_name optc_051 --node_feat_dim 25
```
### Train 

→ Train according to the chosen model:

**Machine 051:**
```bash
python train_link_prediction.py --dataset_name optc_051 --model_name GraphMixer --optimizer Adam --learning_rate 0.0001 --batch_size 48 --num_epochs 1 --gpu 0 --val_ratio 0.065 --test_ratio 0.35 --num_runs 1
```

**Machine 201:**
```bash
python train_link_prediction.py --dataset_name optc_201 --model_name GraphMixer --optimizer Adam --learning_rate 0.0001 --batch_size 48 --num_epochs 1 --gpu 0 --val_ratio 0.098 --test_ratio 0.484 --num_runs 1
```

### Evaluation

→ Evaluate the models by executing the same commands, just replace `train_link_prediction` with `evaluate_link_prediction`.

To obtain results with different temperatures, you can add `--temperature 1` (1 baseline value, tests were conducted with 0.5, 1, 3).

## Tested Models:
- TGAT
- GraphMixer
- DyGFormer

**Notes:**
- EdgeBank has no interest here as it learns nothing.
- Memory-based models are too expensive and could not be run (TGN, DyRep, JODIE).
- CAWN and TCL were not tested.

Voir le env.yml pour lancer les scripts. 
