import numpy as np
import scipy.special as special
import math

from lookup_tables import WI_TABLE, WC_TABLE

# ==============================================================================
def heaviside(x):
  if x < 0:
      y = 0
  else:
      y = 1
  return y
# ==============================================================================

# ==============================================================================
def kronecker(x,y):
  if x == y:
      z = 1
  else:
      z = 0
  return z
# ==============================================================================

# ==============================================================================
def eulerian(n,q):
  A = 0
  for j in range(q+1):
      A = A + (-1)**j*int(special.binom(n+1,j))*(q-j+1)**n
  return A
# ==============================================================================

# ==============================================================================
def M0(N,T):
  M = int(0.5*(N+T+1))
  return M
# ==============================================================================

# ==============================================================================
def W0(N,M):
  W = eulerian(N,M-1)
  return W
# ==============================================================================

# ==============================================================================
def Omega0(N):
  Omega = N
  return Omega
# ==============================================================================

# ==============================================================================
def omegai(N,M):
  if N>1:
      omega = (M-1)*(N-M)+1
  elif N==1:
      omega = 1
  
  return omega
# ==============================================================================

# ==============================================================================
def omegac(N,M):
  if N>1:
      omega = max(2*min(M-1,N-M),1) - kronecker(0.5*(N-1),M-1.0)
  elif N==1:
      omega = 1
  
  return omega
# ==============================================================================

# ==============================================================================
def OmegaI(N):
  Omega = ( 15*N**2 - 2*N**3 -7*N )/6
  return Omega
# ==============================================================================

# ==============================================================================
def OmegaC(N):
  Omega = int(math.floor(((N-1)**2+3)/2))
  return Omega
# ==============================================================================

# ==============================================================================
def findk(N,M,I):

    length=omegai(N,M)
    absmax=0.5*(N-1)*(N-1) - (np.minimum(M-1,N-M))**2
    if M <= 0.5*N:
        Iscore=[(-absmax+2*k) for k in range(length)]
    else:
        Iscore=[(absmax-2*k) for k in range(length)]
    Iscore=np.sort(Iscore) #sort
    Iscore=np.flipud(Iscore)    #reverse
    Iscore=Iscore.tolist() # back to list
    for k in range(length):
        if abs(Iscore[k]-I) < 0.1:
            bestk = k + 1
    #print 'best k = ',bestk
      
    return bestk
# ==============================================================================

# ==============================================================================
def WI(N, M, k):
    # Dynamic formula calculations
    if M == 1 or M == N:
        return 1
    if k == 1:
        return int(special.binom(N - 1, M - 1))
    if M == 2 or M == N - 1:
        return int(special.binom(omegai(N, M) + 1, k)) - 1

    # O(1) Constant-time lookup replacing hundreds of if/elif blocks
    return WI_TABLE.get((N, M, k), -1)
# ==============================================================================

# ==============================================================================

def WC(N, M, D):
    # Dynamic formula calculations first
    if M == 1 or M == N:
        return 1
    if D == 1:
        return 2 * int(special.binom(N - 1, M - 1)) - kronecker(M, 1) - kronecker(M, N)

    om = omegac(N, M)
    if D == 2 and om == 2:
        return W0(N, M) - (
            2 * int(special.binom(N - 1, M - 1))
            - kronecker(M, 1)
            - kronecker(M, N)
        )

    # Check for specific omegac match first, fallback to wildcard (None)
    val = WC_TABLE.get((N, D, om))
    if val is not None:
        return val

    return WC_TABLE.get((N, D, None), -1)
# ==============================================================================

# ==============================================================================
def randomswap(radii):
    
    lastelement = len(radii) - 1
    
    # Choose a random element = swap1
    swap1 = np.random.randint(0,lastelement+1)
    
    # Calculate swap2
    if swap1 == 0:
        swap2 = swap1 + 1
    elif swap1 == lastelement:
        swap2 = swap1 - 1
    else:
        swap2 = swap1 + int(2.0*(np.random.randint(0,2) - 0.5))
    
    # Create the swap array
    swaps = np.array([swap1,swap2])
    #print 'swaps = ',swaps[0],swaps[1]
    
    # Conduct the swap
    newradii=[0 for i in range(len(radii))]
    for i in range(len(newradii)):
        if i == swaps[0]:
            newradii[i] = radii[swaps[1]]
        elif i == swaps[1]:
            newradii[i] = radii[swaps[0]]
        else:
            newradii[i] = radii[i]
    
    return newradii
# ==============================================================================