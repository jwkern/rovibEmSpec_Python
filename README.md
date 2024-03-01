___________________________________________________________________________________________________________________________________________________________________
___________________________________________________________________________________________________________________________________________________________________
___________________________________________________________________________________________________________________________________________________________________
# rovibEmSpec_Python

___________________________________________________________________________________________________________________________________________________________________
GENERAL DESCRIPTION:
This Python script creates the rovibrational spectrum of 12CO assuming the geometry of a rotating disk (i.e. a protoplanetary disk). 

___________________________________________________________________________________________________________________________________________________________________
DATA DESCRIPTION:
The data for this synthetic spectrum are self-contained arrays. 

___________________________________________________________________________________________________________________________________________________________________
CODE DESCRIPTION:
The code uses matplotlib, numpy, math, pylab, and csv Python packages to compute the ro-vibrational spectrum of CO as observed in a rotating disk of gas. 

The code is separated into segments which first initialize variables, arrays, and sets constants. The disk geometry is given in cgs units. The molecular data for CO is read in from 12CO_v10.csv (e.g. transition energies, Einstein coefficients, degeneracies, etc.). The partition function of CO as it relates to temperature is already calculated and also read in from Partfun_12CO16O.csv. 

After this, radiative transfer equations are used to compute the optical depth as a function of wavenumber in cgs, which in turn is used to compute the spectrum. 

The results take a few thousand steps depending on how you set the disk geometry (i.e. the radial and angular steps), and are plotted in a multi-panel plot in which the full spectrum, a single spectral line, T(r), and Sigma(r) are plotted. 

The synthetic spectrum is output to a .csv file. 

___________________________________________________________________________________________________________________________________________________________________
RUNNING THE CODE:
1) Download the python script (rovibEmSpec_JWK.py)
 
2) In a terminal, cd into the directory that now contains the script

3) Run the script by typing the following into the command line:

           python3.8 rovibEmSpec_JWK.py
___________________________________________________________________________________________________________________________________________________________________
___________________________________________________________________________________________________________________________________________________________________
___________________________________________________________________________________________________________________________________________________________________
