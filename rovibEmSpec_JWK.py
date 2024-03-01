import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math
import numpy as np
from numpy import *
import pylab
import csv


##################################################################################
#########   Constants ############################################################
##################################################################################

B = 1.93                                        #rotational constant (cm^-1)
c = 2.99792458e10                               #speed of light (cm*s^-1)
G = 6.67430e-08                                 #gravitational constant (cm^3*g^-1*s^-2)
h = 6.6261e-27                                  #plancks constant (cm^2*g*s^-1)
k_wvn = 0.6950356                               #boltzmann const. (cm^-1*K^-1)
k = 1.3807e-16                                  #boltzmann const. (cm^2*g*s^-2*K^-1)
Msun = 1.989e33                                 #mass of the Sun (g)
molMassCO = 28.01                               #molar mass of CO (g/mol)
NA = 6.0221409e23                               #Avogadro constant (particles/mol)
m_CO=molMassCO/NA                               #mass of CO molecule (g)




##################################################################################
#########   Variables ############################################################
##################################################################################


Mstar = 2*Msun                                  #star's mass
Mdisc = 1e-3*Msun                               #disc mass

Rout = 10*1.4959e13                             #outer radius of emitting region 
Rin = 1*1.4959e13                               #in.rad. of emit.reg (1.49e13cm is 1AU)

T_0=1500                                        #temperature at Rin (K)
T_radExp=-0.5                                   #temperature power law exponent

Sigma_0=1e-8                                    #surface density at Rin (g/cm^2)
Sigma_radExp=-0.5                               #surface density power law exponent

inc_degrees=0                                   #disk inclination in degrees
inc=inc_degrees*np.pi/180                       #disk inclination in radians



##################################################################################
#########  Arrays & Step/Min/Max Values  #########################################
##################################################################################

#radiative transfer arrays and steps##############################################
wl_stepSize=0.001                                #wavelength step size in wvn_cgs 
rndDecimal=3                                     #decimal place to round

wl_max=5.3e-6                                   #maximum wavelength computed
wl_min=4.4e-6                                     #minimum wavelength computed

wvn_max=1/wl_min                                #maximum wavenumber computed
wvn_min=1/wl_max                                #minimum wavenumber computed

wl_steps = int(((wvn_max/100)-(wvn_min/100))/wl_stepSize) #number of wl steps

Bs = np.zeros(wl_steps)                         #blackbody function array
Ls = np.zeros(wl_steps)                         #individual line luminosity array
Ltotal = np.zeros(wl_steps)                     #total luminosity array/spectrum
Lnorm = np.zeros(wl_steps)                      #cont. normalized lum. array
#wl = np.linspace(wl_min, wl_max, wl_steps)      #wavelength array
#wl_microns = np.zeros(wl_steps)                 #micron wavelength array
wvn = np.linspace(wvn_min, wvn_max, wl_steps)   #wavenumber array
wvn_cgs = np.zeros(wl_steps)                    #inverse cm wvn array
gaussProf = np.zeros(wl_steps)                  #empty gaussian profile array
tau_CO12_v10_indProf=np.zeros(wl_steps)         #Optical depth array for single line
tau_CO12_v10_spec=np.zeros(wl_steps)            #Optical depth array for whole spec.

#unit conversions
for i in range(0, wl_steps):
    #wl_microns[i] = wl[i]*1e6
    wvn_cgs[i] = round(wvn[i]/1e2, rndDecimal)




#disk geometry arrays and steps###################################################
R_stepSize=2*1.4959e13                          #step size for radius
R_steps = int((Rout-Rin)/R_stepSize)            #number of radius annuli
R=np.linspace(Rin,Rout,R_steps)                 #radius array

Tgas=np.zeros(R_steps)                          #emtpy temperature array
Q=np.zeros(R_steps)                             #empty Q array
Sigma=np.zeros(R_steps)                         #empty surface density array
Ntot=np.zeros(R_steps)                          #empty number of absorbers array
vrad=np.zeros(R_steps)                          #empty velocity array for v(r)
dA=np.zeros(R_steps)                            #empty diff. area array 

vdisc=[]                                        #empty velocity array for whole disk

bbfs=[]                                         #empty blackbody function array

COdoppler=[]                                    #empty Doppler shift array
wvnDoppler=[]                                   #empty wvn array

theta_stepSize=np.pi/4                          #step size for angles in disk
theta_steps=int(2*np.pi/theta_stepSize)         #number of angular steps
theta=np.linspace(0,2*np.pi-theta_stepSize,theta_steps)        #empty theta array



##################################################################################
#########  Read in CO data #######################################################
##################################################################################

#read in 12CO wvns, A, E, vup, vlow, Jup, Jlow, gup, glow, ID
COdata = open('/home/jwkern/Research/Exopl/Repository/CO_data/12CO_v10.csv')
CO12v10=csv.reader(COdata)
rows=[]
for row in CO12v10:
    rows.append(row)

CO12_v10_wvn_cgs=[]
CO12_v10_A=[]
#CO12_v10_Eup=[]
CO12_v10_Elow=[]
CO12_v10_Jup=[]
CO12_v10_Jlow=[]
CO12_v10_gup=[]
CO12_v10_glow=[]
CO12_v10_ID=[]


for i in range(0,len(rows)):
    CO12_v10_wvn_cgs.append(round(np.double(rows[i][0]),rndDecimal))
    CO12_v10_A.append(np.double(rows[i][1]))
    #Eup = np.double(rows[i][0]) + np.double(rows[i][2])
    #CO12_v10_Eup.append(round(Eup,rndDecimal)) 
    CO12_v10_Elow.append(round(np.double(rows[i][2]),rndDecimal))
    CO12_v10_Jup.append(np.double(rows[i][5]))
    CO12_v10_Jlow.append(np.double(rows[i][6]))
    CO12_v10_gup.append(np.double(rows[i][7]))
    CO12_v10_glow.append(np.double(rows[i][8]))
    CO12_v10_ID.append(str(rows[i][9]))


#CO12_v10_wl_microns=[]
#CO12_v10_wvn=[]
#for i in range(0,len(CO12_v10_wvn_cgs)):
    #wl1 = 1/CO12_v10_wvn_cgs[i]*10000
    #wvn1 = CO12_v10_wvn_cgs[i]*100
    #CO12_v10_wl_microns.append(wl1)
    #CO12_v10_wvn.append(round(wvn1,rndDecimal))

y_test=np.zeros(len(CO12_v10_wvn_cgs))



#read in CO partition function values for T=[0,9000]
CO_Qdata = open('/home/jwkern/Research/Exopl/Repository/CO_data/PartitionFunc/Partfun_12CO16O.csv')
CO12v10_Q=csv.reader(CO_Qdata)
rows_Q=[]
for row in CO12v10_Q:
    rows_Q.append(row)

CO12_v10_QT=[]
CO12_v10_Q=[]

for i in range(0,len(rows_Q)):
    CO12_v10_QT.append(np.double(rows_Q[i][0]))
    CO12_v10_Q.append(np.double(rows_Q[i][1]))






###################################################################################
######### Optical Depth, RTE, and Blackbody Functions #############################
###################################################################################

#setup disk radial profiles
for j in range(0,R_steps):
    Tgas[j]=T_0*(R[j]/Rin)**T_radExp
    Q[j]=CO12_v10_Q[CO12_v10_QT.index(int(Tgas[j]))]
    Sigma[j]=Sigma_0*(R[j]/Rin)**Sigma_radExp
    #dA[j]=R[j]*R_stepSize*theta_stepSize
    Ntot[j]=Sigma[j]/m_CO
    vrad[j]=np.sqrt(G*Mstar/R[j])
    vdisc.append(vrad[j]*np.sin(theta)*np.sin(inc))
    bbfs.append((2.0*h*c**2*wvn_cgs**3)/(np.e**((h*c*wvn_cgs)/(k*Tgas[j])) - 1.0)*((2*np.pi*R[j]*R_stepSize) + (np.pi*(R_stepSize**2))))

#calculate cross sections for each transition
num=np.multiply(CO12_v10_gup,CO12_v10_A)
denom=8*np.pi*(np.multiply(np.multiply(CO12_v10_wvn_cgs,CO12_v10_wvn_cgs),CO12_v10_glow))
crsSecAmp=num/denom     #scalar

#calculate normalized gaussian amplitudes for each transition at every radii
inv_wvn=np.divide(1,CO12_v10_wvn_cgs)
inv_thermo=np.divide(1,np.sqrt(2*np.pi*k*Tgas/m_CO))
gaussNormAmp = np.outer(inv_wvn,inv_thermo)   #2D matrix (rows: transitions, col: radii)

#calculate the number density amplitude for each transition at every radii
NQ=np.divide(Ntot,Q)
NQg=np.outer(CO12_v10_glow,NQ)
inv_thermo2=np.divide(1,Tgas)/k_wvn
ex=np.e**(-np.outer(CO12_v10_Elow,inv_thermo2))
numDenAmp=np.multiply(NQg,ex) #2D matrix (rows: transitions, col: radii)

tau1=[]
tau3=[]
ones3=[]
Tgas3=[]
R3=[]
#generate an amplitude for each transition at each point in the disk 
for i in range(0,len(crsSecAmp)):
    tau1.append(crsSecAmp[i]*numDenAmp[i][:])

tau2=np.multiply(tau1,gaussNormAmp)
tau2=tau2.transpose()
thetadim=np.ones(theta_steps)
raddim=np.ones(R_steps)
wavdim=np.ones(len(CO12_v10_wvn_cgs))

ones2=np.outer(raddim,wavdim)

#generate Gaussians at each wavenumber of every transition
vcratio=np.divide(vdisc,c)
dopShift0=np.add(1,vcratio)
dopShift=np.divide(1,dopShift0)
for j in range(0,R_steps):
    COdoppler.append(np.outer(CO12_v10_wvn_cgs,dopShift[j])) #3D matrix
    tau3.append(np.outer(tau2[j],thetadim))
    ones3.append(np.outer(ones2[j],thetadim))
    Tgas3.append(ones3[j]*Tgas[j])
    R3.append(ones3[j]*R[j])
inv_thermo3=np.divide(m_CO*c**2,2*k*Tgas)
inv_COdopplerSq=np.divide(1,COdoppler)*np.divide(1,COdoppler)

inv_thermo4=[]
for j in range(0,R_steps):
    inv_thermo4.append(inv_thermo3[j]*inv_COdopplerSq[j][:][:])

#inv_thermo4 = [inv_thermo3[j]*inv_COdopplerSq[j][:][:] for j in range(0, R_steps)]

COdoppler_lin=np.array(list(np.array(COdoppler).flat))
inv_thermo4_lin=np.array(list(np.array(inv_thermo4).flat))
tau3_lin=np.array(list(np.array(tau3).flat))
Tgas3_lin=np.array(list(np.array(Tgas3).flat))
R3_lin=np.array(list(np.array(R3).flat))




#create Gaussian profiles and calculate their amplitude at each wavelength
for a in range(0,len(COdoppler_lin)):
    print('Step' + str(a))
    tau_nu = tau3_lin[a]*np.e**(-inv_thermo4_lin[a]*((wvn_cgs-COdoppler_lin[a])**2))
    bbfs = (2.0*h*c**2*wvn_cgs**3)/(np.e**((h*c*wvn_cgs)/(k*Tgas3_lin[a])) - 1.0)*((2*np.pi*R3_lin[a]*R_stepSize) + (np.pi*(R_stepSize**2)))
    Ltotal = Ltotal + bbfs*(1 - np.e**(-tau_nu))



x_test=np.zeros(len(wvn_cgs))

for i in range(0,len(wvn_cgs)-1):
    x_test[i]=10000.0/wvn_cgs[i]
    Ltotal[i]=(Ltotal[i]/11e30)+1

###################################################################################
##########   Plot Details #########################################################
###################################################################################

fig = plt.figure(constrained_layout=True, figsize=(8, 6))
spec = gridspec.GridSpec(ncols=3, nrows=2, figure=fig)


#T vs. R
ax1 = fig.add_subplot(spec[0,0])
plt.plot(R/1.4959e13, Tgas, 'k-')
plt.xlabel('Radius (AU)')
plt.ylabel('Tgas (K)')

#Sigma vs. R
ax2 = fig.add_subplot(spec[0,1])
plt.plot(R/1.4959e13, Sigma, 'k-')
plt.xlabel('Radius (AU)')
plt.ylabel('$\Sigma$ (g/$cm^{2}$)')

#individual line 
ax3 = fig.add_subplot(spec[0,2])
plt.plot(wvn_cgs, Ltotal, 'k-')
#plt.plot(x_test,y_test, 'c.', label = '12CO, v=1-0')
plt.xlabel('Wavenumbers ($cm^{-1}$)')
plt.ylabel('Luminosity (ergs*$s^{-1}$)')
plt.xlim(2103.20,2103.35)
plt.ylim(0.9, 1.35)

#full spectrum
ax4 = fig.add_subplot(spec[1,:])
plt.plot(x_test, Ltotal, 'k-') 
#plt.plot(x_test,y_test, 'c.', label = '$^{12}$CO, v=1-0')
plt.legend(loc='upper left')
plt.xlabel('Wavelength ($\mu$m)')
plt.ylabel('Normalized Flux')#Luminosity (ergs*$s^{-1}$)')
plt.title('CO Ro-vibrational Spectrum Detectable with iSHELL')
plt.xlim(4.45,5.275)
plt.ylim(0.9,1.35)
plt.show()


fbig = wvn_cgs
spec = Ltotal

output = column_stack((fbig,spec))
np.savetxt('COspec.dat',output)
