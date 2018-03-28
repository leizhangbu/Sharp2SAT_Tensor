#11-14-2017
# 2SAT problems: counting Solutions
# LZ
# alpha=1.5 based on 3-regular graph no negation


####################################################
from numpy import *
from numpy.linalg import svd
from numpy.random import random,randint
from scipy.sparse.linalg import svds
import pandas as pd
from random import *
from igraph import *
from bandwidth import *
import time
br = binary_repr

def main():
    Nbits = 10 # number of variables
    alpha = 1.5 # ratio of Nclauses/Nbits  alpha = 1.5 for regular 3 graph
    Nclause = int(Nbits*alpha) # number of clauses
    Ncontraction = int((Nbits-2)/2)  # number of contractions
    nsample = 1000
    Max_data=[]
    fm=open("Maxbond_all.d","w",0)  # maximum bond dimention for every sample
    fms=open("Maxbond_differet_steps.d","w",0) # maximum bond dimention at different contraction step
    f_time=open("Time_to_Sol.d","w",0)   # time to solution
    for i in range (nsample):
        t0=time.time()
        g,max_all=result_RG(Nbits,Nclause,Ncontraction,clean=4)# perform coarse graining algorithm
        Max_data.append(max(max_all))
        f_time.write(str(time.time()-t0))
        f_time.write("\n")
        fm.write(str(max(max_all)))
        fm.write("\n")
        fms.write(str(max_all))
        fms.write("\n")
        fh=open("Numberstats.d","w",0)
        fh.write("Numberfinished")
        fh.write("\n")
        fh.write(str(i))
        fh.close
    f_time.close
    fm.close
    fms.close
    fh=open("MaxBond.d","w")
    fh.write(str(mean(Max_data)))
    fh.write("\n")
    fh.write(str(std(Max_data)))
    fh.close
###################################################
#     define Gate functions  (3 types)            #
#     ID, Blue Circle and Red Square              #
###################################################
def check_boundary(bc,gate):
    if (bc==[-1]*3):
        r=1
    else:
        r=1
        for i in range (len(bc)):
            if (bc[i]>-1):
                if (bc[i]!=gate[i]):
                    r=r*0
    return r

def G_ID(b_in=[-1]*3,b_out=[-1]*3):
    #b_in and b_out : in and out put boundary condition
    g=zeros([2**3,2**3])
    for x in range (0,2):
        for c1 in range (0,2):
            for c2 in range (0,2):
                xp=x
                c1p=c1
                c2p=c2                        
                i=x+c1*2+c2*2**2
                o=xp+c1p*2+c2p*2**2
                o=int(o)
                r1=check_boundary(b_in,[x,c1,c2])
                r2=check_boundary(b_out,[xp,c1p,c2p])
                g[i,:]=r1*g[i,:]
                g[:,o]=r2*g[:,o]
                g[i,o]=r1*r2
    return g

def G_CIR(b_in=[-1]*3,b_out=[-1]*3):
    #b_in and b_out : in and out put boundary condition
    g=zeros([2**3,2**3])
    k=0
    for x in range (0,2):
        for c1 in range (0,2):
            for c2 in range (0,2):
                xp=x
                c1p=abs(abs(x-k)-c1)
                c2p=c2                        
                i=x+c1*2+c2*2**2
                o=xp+c1p*2+c2p*2**2
                o=int(o)
                r1=check_boundary(b_in,[x,c1,c2])
                r2=check_boundary(b_out,[xp,c1p,c2p])
                g[i,:]=r1*g[i,:]
                g[:,o]=r2*g[:,o]
                g[i,o]=r1*r2
    return g
def G_BLOCK(b_in=[-1]*3,b_out=[-1]*3):
    #b_in and b_out : in and out put boundary condition
    g=zeros([2**3,2**3])
    k=0
    for x in range (0,2):
        for c1 in range (0,2):
            for c2 in range (0,2):
                xp=x
                c2p=abs(c2-abs(x-k)*abs(1-c1))
                c1p=abs(c2p-c1)
                i=x+c1*2+c2*2**2
                o=xp+c1p*2+c2p*2**2
                o=int(o)
                r1=check_boundary(b_in,[x,c1,c2])
                r2=check_boundary(b_out,[xp,c1p,c2p])
                g[i,:]=r1*g[i,:]
                g[:,o]=r2*g[:,o]
                g[i,o]=r1*r2
    return g
#########################################################
#               Create Basic Tensors 3 type             #
#               ID, Blue Circle and Red Square          #
#########################################################


def Tensor_ID(b_in,b_out):
    # input Bits and output Bits
    # create a ID tensor, tensor bond Dimension are [2,4,2,4] anti-clockwise from 12 O'clock 
    g=G_ID(b_in,b_out)
    ni=[1,2]
    no=[1,2]
    nL=ni+no
    g=reshape(g,2**array(ni+no),'F')
    t=g
    return t
def Tensor_CIR(b_in,b_out):
    # input Bits and output Bits
    # create a ID tensor, tensor bond Dimension are [2,4,2,4] anti-clockwise from 12 O'clock 
    g=G_CIR(b_in,b_out)
    ni=[1,2]
    no=[1,2]
    nL=ni+no
    g=reshape(g,2**array(ni+no),'F')
    t=g
    return t

def Tensor_BLOCK(b_in,b_out):
    # input Bits and output Bits
    # create a ID tensor, tensor bond Dimension are [2,4,2,4] anti-clockwise from 12 O'clock 
    g=G_BLOCK(b_in,b_out)
    ni=[1,2]
    no=[1,2]
    nL=ni+no
    g=reshape(g,2**array(ni+no),'F')
    t=g
    return t

###################################################################
#                Compression Decimation SVD algorithm             #
###################################################################
def tensor_pairwise_contraction_svd(t1,t2,d1,d2,bd,eps=1e-4):
    """ Performs contraction and SVD of tensors t1 and t2 along the
        bond that connects the respective dimensions d1 and d2. The
        dimension of the contracted bond is truncated after the SVD
        based on the number of nonzero normalized singular values s.
        Finiteness is determined by the condition s/s[0] > eps. """
    s1,s2 = array(shape(t1)),array(shape(t2)) # Get tensor shapes
    t = tensordot(t1,t2,[[d1],[d2]])          # Perform contraction
    # Fold tensor into matrix and perform SVD
    q1,q2 = prod(s1)/s1[d1],prod(s2)/s2[d2]
    tm = reshape(t,[q1,q2],'F')
    u,s,v = svd(tm,0)
    # Get new bond dimension
    ub = len(s[s/s[0]>eps])
    bd=ub
    l1,l2 = list(s1),list(s2)
    l1.pop(d1)
    l2.pop(d2)
    # Obtain new tensors and update network
    ss = sqrt(s)[:ub]
    t1 = reshape((u[:,:ub]*ss),l1+[ub],'F')
    t2 = reshape((v[:ub,:].T*ss),l2+[ub],'F')
    t1,t2 = rollaxis(t1,-1,d1),rollaxis(t2,-1,d2)
    return t1,t2,bd

def tensor_svd(t1,t2,d1,d2,bd,eps=1e-4):
    sh1,sh2 = array(shape(t1)),array(shape(t2)) # Get tensor shapes
    l1,l2=len(sh1),len(sh2)
    t1,t2=rollaxis(t1,d1,l1),rollaxis(t2,d2,l2)
    q1,q2 = prod(sh1)/sh1[d1],prod(sh2)/sh2[d2]
    t1,t2 = reshape(t1,[q1,sh1[d1]],'F'),reshape(t2,[q2,sh2[d2]],'F')
    s1 = svd(t1,0,0)
    ub1 = len(s1[s1/s1[0]>eps])
    if (ub1<min(q1,sh1[d1])):
        u1,s1,v1=svds(t1,ub1)
    else:
        u1,s1,v1=svd(t1,0)
    s2 = svd(t2,0,0)
    ub2 = len(s2[s2/s2[0]>eps])
    if (ub2<min(q2,sh2[d2])):
        u2,s2,v2=svds(t2,ub2)
    else:
        u2,s2,v2=svd(t2,0)
    tt1=tensordot(identity(len(array(s1)))*s1,v1[:len(array(s1)),:],[[1],[0]])
    tt2=tensordot(identity(len(array(s2)))*s2,v2[:len(array(s2)),:],[[1],[0]])
    tt2=rollaxis(tt2,1,0)
    up=array(shape(v1))[1]
    bd=up
    tt1,tt2,bd=tensor_pairwise_contraction_svd(tt1,tt2,1,0,bd,eps)
    tt2=rollaxis(tt2,1,0)
    l1,l2 = list(sh1),list(sh2)
    l1.pop(d1)
    l2.pop(d2)
    t1 = reshape(tensordot(u1[:,:up],tt1,[[1],[0]]),l1+[bd],'F')
    t2 = reshape(tensordot(u2[:,:up],tt2,[[1],[0]]),l2+[bd],'F')
    t1,t2 = rollaxis(t1,-1,d1),rollaxis(t2,-1,d2)

    return t1,t2,bd
    
######################################################
#           SVD on all bonds in tensor Netwokr       #
######################################################
def sweep(g,n=1):
    # sweep methods, n=1 sweep times: 1 sweep = forward and backward
    tn = g.vs["tensor"]   # Tensor network
    bd = g.es["bdim"]     # Bond dimensions
    otoi=g.es["otoi"]
    tn_edge=g.get_edgelist() # tensor edgelist
    ne = g.ecount() 
    for i in range (n):
        for j in range (ne):
            t1=tn[tn_edge[j][0]]
            t2=tn[tn_edge[j][1]]
            d1=otoi[j][0]
            d2=otoi[j][1]
            t1,t2,bd[j]=tensor_svd(t1,t2,d1,d2,bd[j])
            tn[tn_edge[j][0]]=t1
            tn[tn_edge[j][1]]=t2
        for k in range (ne):
            j=ne-k-1
            t1=tn[tn_edge[j][0]]
            t2=tn[tn_edge[j][1]]
            d1=otoi[j][0]
            d2=otoi[j][1]
            t1,t2,bd[j]=tensor_svd(t1,t2,d1,d2,bd[j])
            tn[tn_edge[j][0]]=t1
            tn[tn_edge[j][1]]=t2
    return tn, bd
def initial_boundary(g,Nbits,Nclause,n=1):
    '''
    Initialize both free and fixed variables at boundary
    '''
    tn = g.vs["tensor"]   # Tensor network
    for i in range (Nbits):
        t1=tn[i]
        dim=shape(tn[i])[2]
        ta=ones([1,dim])
        tn[i]=tensordot(ta,t1,[[1],[2]])
        tn[i]=rollaxis(tn[i],0,3)

    for i in range (Nclause):
        j=i*Nbits
        t1=tn[j]
        dim=shape(t1)[1]
        ta=ones([1,dim])
        tn[j]=tensordot(ta,t1,[[1],[1]])
        tn[j]=rollaxis(tn[j],0,2)

    for i in range (Nbits):
        j=Nbits*(Nclause-1)+i
        t1=tn[j]
        dim=shape(t1)[0]
        ta=ones([1,dim])
        tn[j]=tensordot(ta,t1,[[1],[0]])
        tn[j]=rollaxis(tn[j],0,1)

    for i in range (Nclause):
        j=Nbits*i+Nbits-1
        t1=tn[j]
        dim=shape(t1)[3]
        ta=ones([1,dim])
        tn[j]=tensordot(ta,t1,[[1],[3]])
        tn[j]=rollaxis(tn[j],0,4)


    
    return tn

##############################################
# Boundary Contraction Algorithm             #
##############################################        

def contraction_tensor(g,Nbits,Nclause):

    tn=g.vs["tensor"] # initialize new tensor
    bd = g.es["bdim"]
    ne=-1
    tn_new=[]
    for i in range (Nclause):
        j=i*Nbits
        t1=tn[j]
        t2=tn[j+1]
        t12=tensordot(t1,t2,[[3],[1]])
        tall=rollaxis(t12,3,1)
        a0=shape(tall)[0]*shape(tall)[1]
        a1=shape(tall)[2]
        a2=shape(tall)[3]*shape(tall)[4]
        a3=shape(tall)[5]
        tall=reshape(tall,array([a0,a1,a2,a3]),'F')
	
        tn_new.append(tall)
        for j in range(Nbits-4):
            tn_new.append(tn[i*Nbits+j+2])
        j=i*Nbits+Nbits-2
        t1=tn[j]
        t2=tn[j+1]
        t12=tensordot(t1,t2,[[3],[1]])
        tall=rollaxis(t12,3,1)
        a0=shape(tall)[0]*shape(tall)[1]
        a1=shape(tall)[2]
        a2=shape(tall)[3]*shape(tall)[4]
        a3=shape(tall)[5]
        tall=reshape(tall,array([a0,a1,a2,a3]),'F')        
        if (Nbits>2):
	        tn_new.append(tall)
            
    return tn_new


        
def contraction_graph(g,Nbits,Nclause):
    if (Nbits>2):
	    gnew=lattice_2D_gategraph(Nbits-2,Nclause)
    else:
	    gnew=lattice_2D_gategraph(Nbits-1,Nclause)
    gnew.vs["tensor"]=contraction_tensor(g,Nbits,Nclause)
    return gnew
def lattice_2D(Nbits,Nclause):
    g=lattice_2D_gategraph(Nbits,Nclause)
    g.vs["tensor"] = init_tensors(g,Nbits,Nclause)
    return g
def adj_to_lattice(adj,Nbits):
    Ncc=0
    Nclause=int(1.5*Nbits)
    la=zeros([Nclause,Nbits])
    for i in range (Nbits):
	    for j in range (Nbits-i):
	        if adj[i][j+i]==1:
		        la[Ncc][i]=1
		        la[Ncc][j+i]=1
		        Ncc=Ncc+1
    return la
def init_tensors(g,Nbits,Nclause):
    # with input and out put boundary 
    ne = g.ecount()               # Total number of edges
    vl = g.get_adjlist()          # Adjacency list of vertices
    el = g.get_inclist()          # Incidence list of vertices
    nb=5
    if ( ne == 0 ):
        print ("WARNING: init_tensors: gategraph has no edges")
        return None
    listNC=list(2*ones(Nbits))
    listb=range(0,Nbits)
    tn = []                       # Instantiate tensor network
    g=Graph.K_Regular(Nbits,3)
    while(not(g.is_connected())):
	    g=Graph.K_Regular(Nbits,3)
    adj=array(g.get_adjacency().data)
    la=adj_to_lattice(adj,Nbits)
    p=frontmin(la.T)
    la=la.T[p].T

    
    for i in range (Nclause):
        k=0
        for j in range(Nbits):
            if(la[i][j]==1 and k==0):
                x=j
                k=k+1
            if(la[i][j]==1 and k==1):
                y=j   
        for j in range (Nbits):
            if (j==0):
                bi=[-1,0,0]
            else:
                bi=[-1,-1,-1]
            if (j==Nbits-1):
                bo=[-1,1,-1]
            else:
                bo=[-1,-1,-1]
            if (j == x):
                t=reshape(G_CIR(bi,bo),2**array([1,2,1,2]),'F')
                tn.append(t)
            if (j == y):
                t=reshape(G_BLOCK(bi,bo),2**array([1,2,1,2]),'F')
                tn.append(t)
            if ((j!= x) and  (j!=y)):
                t=reshape(G_ID(bi,bo),2**array([1,2,1,2]),'F')
                tn.append(t)                            
    return tn


def lattice_2D_gategraph(Nbits,Nclause):
    n = Nbits*Nclause           # Total number of gates
    g = Graph()             # New graph object
    g.add_vertices(n)       # Add graph vertices
    g.vs["state"] = [-1]*n  # Initialize vertex states to unfixed
    for i in range(n):
        p=[mod(i,Nbits),int(i/Nbits)]
        g.vs[i]["x"] = p[0]
        g.vs[i]["y"] = p[1]
    # Add edges of vertex lattice model
    ne = -1
    for i in range(n-1):                    # Skip final one at top right corner
        p=[mod(i,Nbits),int(i/Nbits)]                   # Current position
        if (p[0]<(Nbits-1)):
            j = i+1  # Neighbor at [x+1,y]
            g.add_edges([(i,j)])
            ne = ne+1

            # "otoi" determines which output connects to which input.
            g.es[ne]["otoi"] = array([3,1])
            g.es[ne]["bdim"]=2**2

        if (p[1]<(Nclause-1)):       
            j = i+Nbits  # Neighbor at [x,y+1]
            g.add_edges([(i,j)])
            ne = ne+1

            # "otoi" determines which utput connects to which input.
            g.es[ne]["otoi"] = array([0,2])
            g.es[ne]["bdim"]=2

    return g


#########################################
#  whole coarse graining process        #
#########################################
def result_RG(Nbits,Nclause,Ncontraction,clean=4):

    g=lattice_2D(Nbits,Nclause)
    g.vs["tensor"]=initial_boundary(g,Nbits,Nclause,n=1)
    maxbd=[]
    max_all=[]
    maxbd.append(max(g.es["bdim"]))
    j=0
    k=0
    b_all=sum(g.es["bdim"])
    while (j!= b_all):
        k=k+1
        g.vs["tensor"],g.es["bdim"]=sweep(g,1)
        j=b_all
        b_all=sum(g.es["bdim"])
        maxbd.append(max(g.es["bdim"]))
    for i in range (Ncontraction):
        g=contraction_graph(g,Nbits,Nclause)
        Nbits=Nbits-2
        maxbd=[]
        j=0
        k=0
        b_all=sum(g.es["bdim"])
        while (j!= b_all):
            j=b_all
            k=k+1
            b_all=sum(g.es["bdim"])
            g.vs["tensor"],g.es["bdim"]=sweep(g,1)
            maxbd.append(max(g.es["bdim"]))
        max_all.append(min(maxbd))
    g=contraction_graph(g,Nbits,Nclause)

    j=0
    k=0
    b_all=sum(g.es["bdim"])
    while (j!= b_all):
        k=k+1
        g.vs["tensor"],g.es["bdim"]=sweep(g,1)
        j=b_all
        b_all=sum(g.es["bdim"])
        maxbd.append(max(g.es["bdim"]))
    return g, max_all

if __name__=="__main__":
    main()
