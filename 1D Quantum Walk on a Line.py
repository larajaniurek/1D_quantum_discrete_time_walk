####  1D quantum walk on a line ####



from numpy import *                    # importing required Python packages
from matplotlib.pyplot import * 





N = 100     # number of randoms steps 
P = 2*N+1    # Number of possible positions on line 





#  defining coin states 
coin0 = array([1, 0])  # |0>
coin1 = array([0, 1])  # |1>





# gives outer product  of coin states 
C00 = outer(coin0, coin0)  # |0><0| 
C01 = outer(coin0, coin1)  # |0><1| 
C10 = outer(coin1, coin0)  # |1><0| 
C11 = outer(coin1, coin1)  # |1><1| 




# combine to give coin operator that is used to flip 
# quantum coin in superposition 
C_hat = (C00 + C01 + C10 - C11)/sqrt(2.)





# shift operator moves along left or right depending
# on the value of the coin 

ShiftPlus = roll(eye(P), 1, axis=0)    #  P is position, so this shifts the
ShiftMinus = roll(eye(P), -1, axis=0)  # position left or right 
S_hat = kron(ShiftPlus, C00) + kron(ShiftMinus, C11) # kron gives tensor product 





# this is walk operator 
# which is c_hat (combined coin and )
U = S_hat.dot(kron(eye(P), C_hat)) 




# now we create initial state of system
# walker is at position 0 
# initial coin state is superpoistion

posn0 = zeros(P)    
posn0[N] = 1          # creating initial position matrix 

psi0 = kron(posn0, (coin0+coin1*1j)/sqrt(2))  # initial wavefunction





# we take N steps by applyng the walk operator N times to the initial 
# wavefunction
psiN = linalg.matrix_power(U, N).dot(psi0)   





# now we need to measure the system!
prob = empty(P)           # creating empty matrix to store probabilities 
for k in range(P):
    posn = zeros(P)
    posn[k] = 1     
    M_hat_k = kron( outer(posn,posn), eye(2))    # measuring probability at position k 
    proj = M_hat_k.dot(psiN)
    prob[k] = proj.dot(proj.conjugate()).real




# here we just plot the probabilities 
fig = figure()
ax = fig.add_subplot(111)
P = int(P)
N = int(N)

plot(arange(P), prob)
plot(arange(P), prob, 'o')
loc = range (0, P, int(P/10)) #Location of ticks
loc = np.array(loc)
xticks(loc)
xlim(0, P)
x_lab = range (-N, N+1, int(P/10))
x_lan = np.array(x_lab)
ax.set_xticklabels(x_lab)
xlabel('Position')
ylabel('Probability')

show()             

