# DAE-Liquid-Structure
Denoising auto-encoder network for efficient sampling of liquid strcture


![Imgur Image](https://github.com/moradza/DAE-Liquid-Structure/tree/main/img/dae.png)



## Method
The DAE network employed in this study learns to map a single snapshot RDF and the corresponding thermodynamic state to its temporally averaged (long-time average) RDF of an MD simulation. The DAE network is trained over 12000 distinct Lennard-Jones liquids at various thermodynamic states with 9.6 million single snapshot RDFs. It is important to note that noise is not added to the input RDF as the inherent fluctuations in a single snapshot RDF play the role of noise in training the DAE network, i.e., each of the 800 RDFs of a given system is a noisy version of the temporally averaged RDF of the same system.  

## Requirment
- Tensorflow 1.4 
- Pandas
- numpy 

## Publication
Moradzadeh, Alireza, and Narayana R. Aluru. "Molecular dynamics properties without the full trajectory: A denoising autoencoder network for properties of simple liquids." The journal of physical chemistry letters 10.24 (2019): 7568-7576. [link](https://pubs.acs.org/doi/abs/10.1021/acs.jpclett.9b02820)
