# NOTE: This script can be modified for different pair styles 
# See in.elastic for more info.

# Choose potential
pair_style nnp dir ../../Potentials/Seed${seed} showew no showewsum 100 resetew no maxew 10000000 cflength 1.0 cfenergy 1.0 emap "2:Au"
pair_coeff * * 6.01

# Setup neighbor style
neighbor 2.5 bin
neigh_modify every 2 delay 0 check yes

# Setup minimization style
min_style	     cg
min_modify	     dmax ${dmax} line quadratic

# Setup output
thermo		1
thermo_style custom step temp pe press pxx pyy pzz pxy pxz pyz lx ly lz vol
thermo_modify norm no
