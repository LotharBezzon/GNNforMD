# Simple-GNN-for-MD
A GNN for per-atom force prediction in frames from classical dynamic simulation.
To train the model I simulated liquid water using LAMMPS (all the files to generate the data are in "LAMMPS input directory", while training data is also present in data/).
The aim is to test the model with a much larger number of molecules than the examples in the training set, and also on simulation of different molecules.

A complete description with the results obtained is in "GNNforMD description.pdf", while a few slides in Italian to present the project are in "presentazione_GNNforMD". Some other results and the possibility to explore them can be found in "test.ipynb". Documentation .html files of all the code is in the "documentation" directory.
