# Shared Experience Actor Critic

This repository builds upon [Shared Experience Actor Critic](https://arxiv.org/abs/2006.07169). In this repository, trained agents, experiments, and other files have been included that were part of the experimentation we did for the course Seminar Advanced Deep Reinforcement Learning at Leiden University. 

The trained runs along with their log files are currently not included within this repository. GitHub has problems when uploading these files. Upon request, we can send over these files for completeness! The results of all experiments done in [the paper](https://github.com/Lexpj/ARL-SEAC/ARL.pdf) can be requested.

## Requirements
Check requirements.txt. Tested on Python 3.8.0 with dependencies stated in requirements.txt. Install via `pip install -r requirements.txt`
NOTE: requirements are updated and different than the [original code base](https://github.com/uoe-agents/seac)

## Training - SEAC
To train the agents in the paper, navigate to the seac directory:
```
cd seac
```

And run:

```train
python train.py with <env config>
```

Valid environment configs are: 
- `env_name=Foraging-15x15-3p-4f-v0 time_limit=25`
- ...
- `env_name=Foraging-12x12-2p-1f-v0 time_limit=25` or any other foraging environment size/configuration.
- `env_name=rware-tiny-2ag-v1 time_limit=500` 
- `env_name=rware-tiny-4ag-v1 time_limit=500` 
- ...
- `env_name=rware-tiny-2ag-hard-v1 time_limit=500` or any other rware environment size/configuration.

Training can also be done via the included slurm job files. Please make sure to read the `slurm_notes.txt` before running these, since some parts have to be manually adjusted.

## Evaluation/Visualization - SEAC

To load and render the pretrained models in SEAC, run in the seac directory

```eval
python evaluate.py
```

Other experiments are in the root folder. These can just be called via 

```eval
python [file]
```

Most evaluation files have some form of cache ready, since even evaluation may take some time. Currently, the files produce the plots from cache. However, each experiment can be run by adjusting the code slightly (it has been commented out).

## Citation
The citation to the original papers is as follows:
```
@inproceedings{christianos2020shared,
  title={Shared Experience Actor-Critic for Multi-Agent Reinforcement Learning},
  author={Christianos, Filippos and Sch{\"a}fer, Lukas and Albrecht, Stefano V},
  booktitle = {Advances in Neural Information Processing Systems},
  year={2020}
}
```
The two environments used within these experiments can be cited by:
```
@inproceedings{papoudakis2021benchmarking,
   title={Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks},
   author={Georgios Papoudakis and Filippos Christianos and Lukas Schäfer and Stefano V. Albrecht},
   booktitle = {Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks (NeurIPS)},
   year={2021},
   openreview = {https://openreview.net/forum?id=cIrPX-Sn5n},
}
```

## Contributors
Lex Janssens, MSc Computer Science at Leiden University, s2989344@vuw.leidenuniv.nl
Koen Oppenhuis, MSc Computer Science at Leiden University, s1692836@vuw.leidenuniv.nl
