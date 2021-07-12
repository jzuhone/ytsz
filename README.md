# ytSZ

## Overview

The change in the CMB intensity due to Compton scattering of CMB
photons off of thermal electrons in galaxy clusters, otherwise known as the
Sunyaev-Zeldovich (S-Z) effect, can to a reasonable approximation be 
represented by a projection of the pressure field of a cluster. However, the 
*full* S-Z signal is a combination of thermal and kinetic
contributions, and for large frequencies and high temperatures
relativistic effects are important. For computing the full S-Z signal
incorporating all of these effects, there is a library:
[SZpack](http://www.jb.man.ac.uk/~jchluba/Science/SZpack/SZpack.html)
([Chluba et al 2012](http://adsabs.harvard.edu/abs/2012MNRAS.426..510C)).

`ytSZ` makes it possible to make projections of the full S-Z signal given the 
properties of the thermal gas in the simulation using SZpack. SZpack has 
several different options for computing the S-Z signal, from full
integrations to very good approximations. Since a full or even a
partial integration of the signal for each cell in the projection
would be prohibitively expensive, we use the method outlined in
[Chluba et al 2013](http://adsabs.harvard.edu/abs/2013MNRAS.430.3054C) to 
expand the total S-Z signal in terms of moments of the projected optical 
depth ![tau](images/tau.png), projected electron temperature 
![Te](images/te.png), and velocities ![beta_par](images/beta_par.png) and 
![beta_perp](images/beta_perp.png) (their equation 18):

![expansion](images/expansion.png)

`ytSZ` makes projections of the various moments needed for the
calculation, and then the resulting projected fields are used to
compute the S-Z signal. In our implementation, the expansion is carried out 
to first-order terms in ![Te](images/te.png) and zeroth-order terms in
![beta_par](images/beta_par.png) by default, but terms at higher-order can 
be optionally included.

## Installing

First, install SZpack by downloading version 1.1.1 
[here](https://www.cita.utoronto.ca/~jchluba/SZpack/_Downloads_/SZpack.v1.1.1.tar.gz).
SZpack depends on GSL. 

