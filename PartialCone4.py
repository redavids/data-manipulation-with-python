from sage.all import*
import os
import subprocess
import sys
import time
import copy
#Nardo is Sweet

def merge(list,i,j): #only works when i < j !!!!!
    r = []
    a = list.pop(i)
    b = list.pop(j-1) #index goes down when you take i out
    if type(a) == sage.rings.integer.Integer: #to do get rid of these cases by [i] blocks length 1
        r.append(a)
    elif type(a) == int:
        r.append(a)
    else:
        for k in range(len(a)):
             r.append(a[k])
    if type(b) == sage.rings.integer.Integer:
        r.append(b)
    elif type(b) == int:
        r.append(b)
    else:
        for m in range(len(b)):
             r.append(b[m])
    list.append(r)
    return list

#needed a version of merge that did not change the original object!
def merge2(l,i,j): #only works when i < j !!!!!
    r = []
    list = copy.deepcopy(l) #here on is the same
    a = list.pop(i)
    b = list.pop(j-1) #index goes down when you take i out
    if type(a) == sage.rings.integer.Integer: #to do get rid of these cases by [i] blocks length 1
        r.append(a)
    elif type(a) == int:
        r.append(a)
    else:
        for k in range(len(a)):
             r.append(a[k])
    if type(b) == sage.rings.integer.Integer:
        r.append(b)
    elif type(b) == int:
        r.append(b)
    else:
        for m in range(len(b)):
             r.append(b[m])
    list.append(r)
    return list




#maybe rewrite merge using extend() method
def TwoSets(l):
    sets = []
    for i in range(len(l)):
        r = [[l[i],l[j]] for j in range(i + 1, len(l))]
        while len(r) > 0:
            sets.append(r.pop(0))
    return sets

def covering_relations(l): #this one gives my "canonical chains"
    lnew = []
    while len(l) > 0:
        seedlist = l.pop(0)
        #print seedlist
        seed = seedlist[len(seedlist)-1]
        if len(seed) == 1:
            lnew.append(seedlist)
        else:
            for i in range(len(seed)):
                for j in range(i+1,len(seed)):
                    sprout = copy.deepcopy(seed)
                    merged_blocks = merge(sprout,i,j)
                    merged_blocks = [list(set(merged_blocks[k])) for k in range(len(merged_blocks))]
                    #print merged_blocks
                    merged_blocks.sort()
                    #print merged_blocks
                    update = copy.deepcopy(seedlist)
                    update.append(merged_blocks)
                    #print update
                    lnew.append(update)
    return lnew

def Ucovering_relations(l):
    lnew = []
    while len(l) > 0:
        seedlist = l.pop(0)
        #print seedlist
        seed = seedlist[len(seedlist)-1]
        if len(seed) == 1:
            lnew.append(seedlist)
        else:
            for i in range(len(seed)):
                for j in range(i+1,len(seed)):
                    sprout = copy.deepcopy(seed)
                    merged_blocks = merge(sprout,i,j)
                    #merged_blocks = [list(set(merged_blocks[k])) for k in range(len(merged_blocks))]
                    #print merged_blocks
                    #merged_blocks.sort()
                    #print merged_blocks
                    update = copy.deepcopy(seedlist)
                    update.append(merged_blocks)
                    #print update
                    lnew.append(update)
    return lnew

def test_covering_relations(l):
    lnew = []
    while len(l) > 0:
        seedlist = l.pop(0)
        #print seedlist
        seed = seedlist[len(seedlist)-1]
        if len(seed) == 1:
            lnew.append(seedlist)
        else:
            for i in range(len(seed)):
                for j in range(i+1,len(seed)):
                    sprout = copy.deepcopy(seed)
                    merged_blocks = merge(sprout,i,j)
                    merged_blocks = [list(set(merged_blocks[k])) for k in range(len(merged_blocks))]
                    print merged_blocks
                    #merged_blocks.sort()
                    #print merged_blocks
                    update = copy.deepcopy(seedlist)
                    update.append(merged_blocks)
                    #print update
                    lnew.append(update)
    return lnew



def chains(n): #makes a list of all the chains in the partition lattice
    one = [[[[i + 1] for i in range(n)]]] #this is a list containing the zero element
    for i in range(n-1):
        one = covering_relations(one)
    return one

def Uchains(n):
    one = [[[[i + 1] for i in range(n)]]] #this is a list containing the zero element
    for i in range(n-1):
        one = Ucovering_relations(one)
    return one

def test_chains(n):
    one = [[[[i + 1] for i in range(n)]]] #this is a list containing the zero element
    for i in range(n-1):
        one = test_covering_relations(one)
    return one

def canonical_chains(chain_list):
    C = copy.deepcopy(chain_list)
    fixed_list = []
    while len(C) > 0 :
        x = C.pop(0)
        y = [[[list(set(x[i][j])) for j in range(len(x[i]))]for i in range(len(x))]]
        fixed_list.append(y)
    return fixed_list
        
        

def bonds(t,p,n):
    w = []
    if len(t) == 2:
       ll = []
       for q in range(len(p)):
           for m in range(len(t)):
               if t[m] in p[q]:
                   ll.append(len(p[q]))
       LL = prod(ll)
       w.append(LL)
    else:    
        for j in range(len(t)): #start with the traversal
            l = []
            for r in range(len(p)): #find where it is in the partition
                if type(p[r]) == sage.rings.integer.Integer: #if the partition block size 1
                    for k in range(len(t[j])):
                        if t[j][k] == p[r]:
                            l.append(1)
                else:
                    for k in range(len(t[j])):
                        if t[j][k] in p[r]:
                           l.append(len(p[r]))
            L = prod(l)
            w.append(L)
    return w

def Ray(t,p,n): #traversal, partition, number of taxa/size of partition lattice
    coords = TwoSets([m + 1 for m in range(n)]) #coords of vector space
    w = bonds(t,p,n) #this gives the list of bond strengths for elements of the traversal
    v = []
    for i in range(len(coords)): #add something to v for each coord
        if len(p) == 2: #had to troubleshoot code for traversals that have one pair...see bonds and pairs functions
           if set(t) == set(coords[i]):
               v.append(w[0])
           else:
               v.append(0)
        else:
           if set(coords[i]) in [set(t[k]) for k in range(len(t))]: #check to see if coord in traversal
               for j in range(len(t)): #then check which it is and add the right weight from w
                   if set(t[j]) == set(coords[i]):
                      v.append(w[j])
           else:
               v.append(0) #0 for stuff not in the traversal
    return v

def Rays(c):#chain in partition lattice, number of taxa
    C = [c[j] for j in range(1, len(c)-1)] #get the partitions except 0 and 1 element
    n = len(c[0])
    ray_list = [[1 for k in range((n*(n-1))/2)]] #this avoids my zero element bug for now in traversals function
    while len(C) > 0:
        p = C.pop(0)
        T = traversals(p)
        p_rays = [Ray(T[i],p,n) for i in range(0,len(T))]
        ray_list = ray_list+p_rays
    return ray_list

def Partial_Rays(c):#(partial) chain in partition lattice, number of taxa 
    C = [c[j] for j in range(1, len(c)-1)]  #get the partitions except 0 and 1 element
    n = len(c[0])
    top = c[len(c)-1] #get the top element
    top_elt_pairs = top_pairs(top)
    tuv = unit_vectors(top_elt_pairs,n)
    ray_list = [[1 for k in range((n*(n-1))/2)]] + tuv  #ones vec + stuff from top 
    while len(C) > 0:
        p = C.pop(0)
        T = traversals(p)
        p_rays = [Ray(T[i],p,n) for i in range(0,len(T))]
        ray_list = ray_list+p_rays
    return ray_list

def unit_vectors(l,n): #unit vectors for two elements sets in list l: to use in Partial_Rays
    coords = TwoSets([m + 1 for m in range(n)]) #coords of vector space
    s = copy.deepcopy(l)
    w = [] #list of unit vectors to return
    while len(s) > 0:
         t = s.pop(0)
         v = []
         for i in range(len(coords)): #add something to v for each coord
             if set(t) == set(coords[i]):
               v.append(1)
             else:
               v.append(0)
         w.append(v)
    return w
         
    

    




#There is still a bug with pairs for parititons with two blocks                  
def pairs(p):
    #q = fix(p) #can change this when I adjust the code for the chains
    block_pairs = TwoSets(p)
    if len(block_pairs) == 1: #this happens when p has two blocks this code is awful but it works
        w = block_pairs.pop(0)
        v = [[w[0][i]] + [w[1][j]] for i in range(len(w[0])) for j in range(len(w[1]))]
        return v
    else:
        L = []
        while len(block_pairs) > 0:
            t = block_pairs.pop(0)
            l = [[t[0][i]] + [t[1][j]] for i in range(len(t[0])) for j in range(len(t[1]))] 
            L.append(l) #every l makes the choices for the traversal for that pair of blocks
        return L

def top_pairs(p): #for partial cones!  gets the pairs for the unit vectors for the end of the partial chain
    #q = fix(p) #can change this when I adjust the code for the chains
    block_pairs = TwoSets(p)
    if len(block_pairs) == 1: #this happens when p has two blocks this code is awful but it works
        w = block_pairs.pop(0)
        v = [[w[0][i]] + [w[1][j]] for i in range(len(w[0])) for j in range(len(w[1]))]
        return v
    else:
        L = []
        while len(block_pairs) > 0:
            t = block_pairs.pop(0)
            l = [[t[0][i]] + [t[1][j]] for i in range(len(t[0])) for j in range(len(t[1]))] 
            L = L + l #every l makes the choices for the traversal for that pair of blocks
        return L

def traversals(p): #to fix, still nests the traversal of zero element too much
    B = pairs(p)
    if B == []:
        print "no traversal of maximal partition"
    elif len(p) == 2:
        return B
    else:
        M = B.pop(0)
        N = B.pop(0) #start building the array
        M = [[M[i]] + [N[j]] for i in range(len(M)) for j in range(len(N))]
        while len(B) > 0:
            K = B.pop(0)
            M = [M[m] + [K[k]] for m in range(len(M)) for k in range(len(K))]
        return M
        
def scale(r,w): #'normalizes'the rays into a common hyperplane where coords sum to s; ray r is a list and s is rational
    S = sum(r)
    scaled_ray = [(w*r[i])/S for i in range(len(r))]
    return scaled_ray
    





def action(g,C): #symmetric group element g acts on chain C
    gC = [[[g(C[j][k][i]) for i in range(len(C[j][k]))]for k in range(len(C[j]))]for j in range(len(C))]
    for i in range(len(gC)):
        for j in range(len(gC[i])):
            gC[i][j].sort()
        gC[i].sort()
    return gC

def Uaction(g,C): #symmetric group element g acts on chain C
    gC = [[[g(C[j][k][i]) for i in range(len(C[j][k]))]for k in range(len(C[j]))]for j in range(len(C))]
    #for i in range(len(gC)):
        #for j in range(len(gC[i])):
           # gC[i][j].sort()
        #gC[i].sort()
    return gC

def test_action(g,C):
    gC = [[[g(C[j][k][i]) for i in range(len(C[j][k]))]for k in range(len(C[j]))]for j in range(len(C))]
    for i in range(len(gC)):
        for j in range(len(gC[i])):
            gC[i][j].sort()
        #gC[i].sort()
    return gC



def orbits(chain_list,n): #list of chains in partition lattice on n letters
    l1 = copy.deepcopy(chain_list)
    l2 = [] #for the output
    tracking = [] #put a chain in here whenever you first see it
    ticker1 = 0
    ticker2 = 0
    H = SymmetricGroup(n).list()
    while len(l1) > 0:
        t = l1.pop(0)
        if t in tracking:
           ticker1 = ticker1 + 1
        else:
            W = [] #to hold the orbit
            T = [action(H[i],t) for i in range(len(H))]
            while len(T) > 0:
                r = T.pop(0)
                if r in W:
                    ticker2 = ticker2 + 1
                else:
                    W.append(r)
                    tracking.append(r)
            l2.append(W)
    return l2, tracking, ticker1, ticker2

def Uorbits(chain_list,n): #list of chains in partition lattice on n letters
    l1 = copy.deepcopy(chain_list)
    l2 = [] #for the output
    tracking = [] #put a chain in here whenever you first see it
    ticker1 = 0
    ticker2 = 0
    H = SymmetricGroup(n).list()
    while len(l1) > 0:
        t = l1.pop(0)
        if t in tracking:
           ticker1 = ticker1 + 1
        else:
            W = [] #to hold the orbit
            T = [Uaction(H[i],t) for i in range(len(H))]
            while len(T) > 0:
                r = T.pop(0)
                if r in W:
                    ticker2 = ticker2 + 1
                else:
                    W.append(r)
                    tracking.append(r)
            l2.append(W)
    return l2, tracking, ticker1, ticker2

def test_orbits(chain_list,n): #list of chains in partition lattice on n letters
    l1 = copy.deepcopy(chain_list)
    l2 = [] #for the output
    tracking = [] #put a chain in here whenever you first see it
    ticker1 = 0
    ticker2 = 0
    H = SymmetricGroup(n).list()
    while len(l1) > 0:
        t = l1.pop(0)
        if t in tracking:
           ticker1 = ticker1 + 1
        else:
            W = [] #to hold the orbit
            T = [test_action(H[i],t) for i in range(len(H))]
            while len(T) > 0:
                r = T.pop(0)
                if r in W:
                    ticker2 = ticker2 + 1
                else:
                    W.append(r)
                    tracking.append(r)
            l2.append(W)
    return l2, tracking, ticker1, ticker2

def unique(list):
    track = 0
    p = []
    for i in range(len(list)):
        if list[i] in p:
            track = track + 1
        else:
            p.append(list[i])
    return p, track
                
#here is code with slow upgma by Ruth ... R-UPGMA
def RUPGMA(d,n): #takes distance vector length (n*(n-1))/2as input, n number of taxa
    zero = [[[i + 1] for i in range(n)]] 
    D = copy.deepcopy(d)
    #print D
    while len(zero)< n-1:
        coords = TwoSets([m for m in range(len(zero[-1]))]) #coords of vector space NOT IN TAXA NAMES THIS TIME!!!
        y = min(D)
        u = D.index(y) #get position of smallest
        a = coords[u][0]
        b = coords[u][1]
        #print a,b
        M = merge2(zero[-1],a,b) #call merge2 on the min distance blocks
        D = distances(M,d,n) #new distances vector
        zero = zero + [M]
        #print M,D,zero
    for r in range(len(zero)): #make canconical form
        for s in range(len(zero[r])):
            zero[r][s].sort()
        zero[r].sort()
    zero = zero + [[[i+1 for i in range(n)]]] #just add the top at the end
    return zero
    
#this makes more computations for each plae in the loop.. BAD
def distances(partition,d,n): #does not depend on where you are in the algorithm
    coords = TwoSets(partition)
    dcoords = TwoSets([m + 1 for m in range(n)])
    D = [] #for output vector
    for i in range(len(coords)):
        A = len(coords[i][0])
        B = len(coords[i][1])
        #print A,B
        a = float(1)/A
        b = float(1)/B
        #print a,b
        S = coords[i][0] + coords[i][1]
        T = TwoSets(S)
        #print T
        W = [set(T[r]) for r in range(len(T))]
        #print W
        avg_list = []
        for k in range(len(dcoords)):
            if set(dcoords[k]) in W:
                avg_list.append(d[k])
        #print avg_list
        Out = a*b*sum(avg_list)
        #print Out
        D.append(Out)
    return D


def Umerge(p,i,j):
    l = copy.deepcopy(p)
    r = []
    a = l.pop(i)
    b = l.pop(j-1) #index goes down when you take i out
    if type(a) == sage.rings.integer.Integer: #to do get rid of these cases by [i] blocks length 1
        r.append(a)
    elif type(a) == int:
        r.append(a)
    else:
        for k in range(len(a)):
             r.append(a[k])
    if type(b) == sage.rings.integer.Integer:
        r.append(b)
    elif type(b) == int:
        r.append(b)
    else:
        for m in range(len(b)):
             r.append(b[m])
    l.append(r)
    return l

# note in Ustep indexing of partition blocks must match matrix indices
def Ustep(M,p): #M is symmetric distance matrix with infty on diagonal, p is partition
    q = copy.deepcopy(p)
    N = len(M.rows())
    B = matrix([[float(M[m][n]) for n in range(len(M[m]))] for m in range(len(M.rows()))])
    Find = [[float(M[m][n]) for n in range(len(M[m]))] for m in range(len(M.rows()))]
    Find = flatten(Find)
    minplace = Find.index(min(Find))
    Row = minplace // N
    Column = Mod(minplace, N)
    L = B.rows()
    LL =  [Row, Column]
    LL.sort()
    #print LL
    [i,j] = LL
    vi = list(L.pop(i)) #turn to list to call pop on it
    vj = list(L.pop(j-1)) #deleting the rows of matrix for i,j but we need vi, vj later
    #print vi, vj
    vi.pop(i)
    vi.pop(j-1) #get rid of distance pair and infty entry for these vectors
    #print vi
    vj.pop(i)
    vj.pop(j-1)
    #print i,j,vi,vj
    Z = matrix(L) #now we get rid of i,j columns
    #print Z
    T = Z.columns()
    T.pop(i) #fix (sorting issue)
    T.pop(j-1) 
    W = matrix(T).transpose() #need to transpose due to sage matrix constructor
    #print W
    w = [] # new vector of distances
    li = len(q[i])
    lj = len(q[j])
    r = Umerge(q,i,j)
    for k in range(len(r)-1):
        u = (li/(float(li + lj)))*vi[k] + (lj/(float(li + lj)))*vj[k]
        w.append(u)
    #print w
    U = matrix([w]).transpose()
    #print U
    W = W.augment(U) 
    #print W
    w.append(100000)
    #print w
    V = matrix([w])
    #print V
    W = W.stack(V)
    #print W
    return W,r

def reformat(l,n):
    m = matrix(RDF, n, n)
    ll = copy.deepcopy(l)
    for k in range(n):
        for i in range(n):
            if i < k:
                m[k,i] = 0
            elif i == k:
                m[k,i] = 0
            else:
                m[k,i] = ll.pop(0) #want to use pop to ensure error is raised if the matrix/list dimensions fight
    return m 

def re_reformat(M):
    n = len(M.rows())
    m = matrix(RDF, n, n)
    for k in range(n):
        for i in range(n):
            if i > k:
                m[i,k] = M[k,i]
            elif i == k:
                m[i,k] = 100000
            else:
                m[i,k] = M[i,k]
    if m.is_symmetric() == True:
        return m
    else:
        print 'error: matrix not symmetric'

    
def UPGMA(l,n):
    M = re_reformat(reformat(l,n))
    one = [[i+1 for i in range(len(M.rows()))]]
    p = [[i+1] for i in range(len(M.rows()))]
    C = [p]
    while len(p) > 2:
        M,p = Ustep(M,p)
        #print M,p
        C = C + [p]
    C = C + [one]
    for r in range(len(C)-1): #make canonical form
        for s in range(len(C[r])):
            C[r][s].sort()
        C[r].sort()
    C[-1].sort()
    return C

def SUPGMA(l,n): #unordered version
    M = re_reformat(reformat(l,n))
    one = [[i+1 for i in range(len(M.rows()))]]
    p = [[i+1] for i in range(len(M.rows()))]
    C = [p]
    while len(p) > 2:
        M,p = Ustep(M,p)
        #print M,p
        C = C + [p]
    C = C + [one]
    return C





def sample2(simplex):
    w = len(simplex)
    U = RealDistribution('uniform',[0,1])
    r = [U.get_random_element() for i in range(w-1)] #cuts [0,1] into len(simplex) pieces
    #print r
    r.sort()
    #print r
    s = [r[0]] + [r[i + 1] - r[i] for i in range(w-2)]+[1-r[-1]] #list of convex coeffs
    #print s
    W = [[s[i]*simplex[i][j] for j in range(len(simplex[i]))] for i in range(len(simplex))]
    #print W
    sample_point = [sum([W[i][j] for i in range(len(W))]) for j in range(len(W[0]))] #Double Check this later
    return sample_point
    
    
    
#HEY YOU NEED TO FIX WHERE!!  HOW MANY SUBINTERVALS DOES S DIVIDE [0,1] INTO!!!!!!!!

def where(k,S): #get k points from [0,totalflatvol] subdivided into subints with left endpoints S
    totalflatvol = S[-1] #S is real_estate, last entry is totalfflatvol
    U = RealDistribution('uniform',[0,totalflatvol])
    points = [U.get_random_element() for i in range(k)]
    #print points
    places = [0 for j in range(len(S))] #the list of how many points in each subint: initialize to zero for each index
    a = 0
    b = 0 
    while len(points) > 0:
        point = points.pop(0)
        if point > S[-1]:
            a = a + 1
            places[-1] = places[-1] + 1
        elif point < S[1]:
            b = b + 1
            places[0] = places[0] + 1
        else:
            Test = False
            y = 0
            while Test == False: #make y go up until the point is past S[y]
                if point > S[y + 1]:
                   y = y + 1
                else:
                   Test = True #change to true the first time the point is not greater than the next divider
            places[y] = places[y] + 1 #add the count of the point to the list
            #print y
    return places #,a,b

def hits(k,S): #get k points from [0,totalflatvol] subdivided into subints with left endpoints S
    totalflatvol = S[-1] #S is real_estate, last entry is totalfflatvol
    U = RealDistribution('uniform',[0,totalflatvol])
    points = [U.get_random_element() for i in range(k)]
    #print points
    places = [0 for j in range(len(S))] #the list of how many points in each subint: initialize to zero for each index
    while len(points) > 0:
        point = points.pop(0)    
        Test = False
        y = 0
        while Test == False: #make y go up until the point is past S[y]
            if point > S[y]:
               y = y + 1
            else:
               Test = True #change to true the first time the point is not greater than the next divider
        places[y] = places[y] + 1 #add the count of the point to the list
        #print y
    return places 


def weight_maker(T): #T is my_format triangulation, fixed 5/7
    real_estate = [0]
    d = len(T[0])-1 #d is 1 less than number of points in each simplex
    for i in range(len(T)):
        gens = T[i]
        #print gens
        newmatrix = []
        for j in range(1,d +1):
            newmatrix.append([gens[j][k]-gens[0][k] for k in range(d+1)])
        newmatrix = matrix(newmatrix).transpose()
        #print newmatrix
        if rank(newmatrix) == d: #we only add real estate for 'real simplices'???
            real_estate.append(real_estate[-1] + float(sqrt(det(newmatrix.transpose()*newmatrix))/factorial(d)))
        else:
            real_estate.append(real_estate[-1])
    #print real_estate
    #z = real_estate[-1]
    #y = 1/z
    #real_estate = [y*real_estate[m] for m in range(len(real_estate))] #my normalizing was bunk! use huggins method
    #print real_estate
    real_estate.pop(0)
    if len(real_estate) == len(T):
        return real_estate
    else:
        print 'error: number of subintervals does not equal number of simplices'

def tri_reader(name): #name is a string like 'TEST.txt' to rep a file
    H = open(name, 'r')
    S = H.read()
    H.close()
    S = S.replace('{','[')
    S = S.replace('}',']')
    S = S.replace('\n',' ')
    S = S.replace(' ',',')
    s = '[' + S + ']'
    T = eval(S)
    return T

def polymake_reader(filename): #name is polymake file like 'mypoints.pc'
    H = open(filename, 'r')
    S = H.read()
    print 'read the file'
    H.close()
    i1 = S.find('FACETS')
    S = S[i1:]
    #print S[0:100]
    i2 = S.find('<v>') #the first place simplices start
    S = S[i2:]
    #print S[0:100]
    i3 = S.rfind('</v>') + 4 #the end of the simplices
    S = S[:i3]
    #print S[len(S)-100:]
    S = S.replace('<v>','[')
    S = S.replace('</v>',']')
    #print S[0:300]
    S = S.replace('\n          ',' ')
    #print S[0:400]
    S = S.replace(' ',', ')
    S = '[' + S + ']'
    #print S[0:400]
    #print S[len(S)-100:]
    Tri = eval(S)
    return Tri

def bypass_reader(X,filename, outname): #name is polymake file like 'mypoints.pc'
    H = open(filename, 'r')
    S = H.read()
    print 'read the file'
    H.close()
    i1 = S.find('FACETS')
    S = S[i1:]
    print S[0:100]
    i2 = S.find('<v>') #the first place simplices start
    S = S[i2:]
    print S[0:100]
    i3 = S.rfind('</v>') + 4 #the end of the simplices
    S = S[:i3]
    print S[len(S)-100:]
    S = S.replace('<v>','[')
    S = S.replace('</v>',']')
    print S[0:300]
    S = S.replace('\n          ',' ')
    print S[0:400]
    S = S.replace(' ',', ')
    S = '[' + S + ']'
    print S[0:400]
    print 'hi'
    print S[len(S)-100:]
    W = ''  #empty string
    for j in range(len(S)):
        if RepresentInt(S[j]) == False:
            W = W + S[j]
        else:
            W = W + str(X[int(S[j])])
    print W[:400]
    W = W.replace('[','{')
    W = W.replace(']','}')
    print W[:400]
    print 'hi'
    print W[len(W)-100:]
    G = open(outname,'w')
    G.write(W)
    G.close()
    return W
    print 'done'
    
        

def RepresentInt(s): #hack to look for integers in the string
    try:
        int(s)
        return True
    except ValueError:
        return False
    
def mathematica_triangulation(X,filename, outname): #outname is who we look for to load into mathematica
    Temp = polymake_reader(filename) #note polymake filename usually ends in pc
    Y = my_format(X,Temp)
    A = str(Y)
    A = A.replace('[','{')
    A = A.replace(']','}')
    G = open(outname,'w')
    G.write(A)
    G.close()
    return A
    print 'done'
    


def partial_dist(T,p,w,n): #Not matlab input!!! T is a list of lists of vectors, not just indices... n=#taxa, w=real_estate p=#samples
    Out = {} #dict where keys are trees and values are number of points out of p that UPGMA gives the tree
    Y = hits(p,w) #replaced flawed "where" on 5/8, tells you how many of the p points to take from each simplex in T
    totalvolscaler = float(w[-1]/p)
    Volumes = {} #dict where keys are trees and values are the sum of contribution(i) 
    #print Y
    r = 0
    tracker = [] #gather reps for dist for error test
    for i in range(len(Y)):
        if Y[i] == 0: 
            r = r + 1 #keeps track of how many simplices get no hits
        else:
            inputs = [sample2(T[i]) for k in range(Y[i])] #uniformly sampled points from T[i], Y[i] tells you how many
            trees = [str(UPGMA(inputs[j],n)) for j in range(Y[i])] #run UPGMA on the Y[i] points
            #print trees
            for m in range(len(trees)): 
                tree = trees[m]
                if tree in Out: #if tree already appears as a key in the dictionary
                    Out[tree] = Out[tree] + 1 #indicates one more point went to that tree
                    Volumes[tree] = Volumes[tree] + JacobianMultiplier(T[i],inputs[m]) #add contribution(i) to tree's entry in Volumes
                else: #if first time we see the tree
                    #tracker.append([inputs[m], str(UPGMA(inputs[m], n))])
                    Out.update({tree:1}) #make a key for this tree and add 1
                    Volumes.update({tree:JacobianMultiplier(T[i],inputs[m])}) #make a key for this tree and add the contribution(i)
    return Out,r,Y, Volumes, totalvolscaler #tracker

def my_normal_format(X,T): #can put this inside a function later makes a good T for partial_dist, T is polymake triangulation in python 
    Y = []
    for m in range(len(X)): #this "normalizes the points to the sphere" as in Huggins code
        u = 1/(float(sqrt(sum([X[m][n]**2 for n in range(len(X[m]))])))) #regular old 2-norm
        print u
        U = [u*X[m][t] for t in range(len(X[m]))]
        print U
        Y.append(U)
    Out = [] 
    for i in range(len(T)):
        Out.append([Y[T[i][j]] for j in range(len(T[i]))]) #gives each simplex as list of points in Y where Y is normalized X
    return Out

def my_format(X,T): #can put this inside a function later makes a good T for partial_dist, T is polymake triangulation in python 
    #Y = []
    #for m in range(len(X)): #this "normalizes the points to the sphere" as in Huggins code
        #u = 1/(float(sqrt(sum([X[m][n]**2 for n in range(len(X[m]))])))) #regular old 2-norm
        #print u
        #U = [u*X[m][t] for t in range(len(X[m]))]
        #print U
        #Y.append(U)
    Out = [] 
    for i in range(len(T)):
        Out.append([X[T[i][j]] for j in range(len(T[i]))]) #gives each simplex as list of points in Y where Y is normalized X
    return Out

def JacobianMultiplier(X,z): #X is simplex in my_format triangulation, z is point for partial_dist, z was taken from X
    d = len(X)-1
    #print d
    gens = []
    for j in range(1,d +1):
        gens.append([X[j][k]-X[0][k] for k in range(d+1)])
    #print gens
    gens = matrix(gens).transpose()
    #print gens
    if rank(gens) < d: #we only add real estate for 'real simplices'???
        return 0
    else:
        oldvol = det(gens.transpose()*gens)
        #print oldvol
        normdivisor = 1/float(sqrt(sum([z[i]**2 for i in range(len(z))])))
        #print normdivisor
        Z = matrix([z])
        Y = (Z.transpose()*Z)*(normdivisor**2)
        #print Y
        I = identity_matrix(RDF,d + 1)
        #print I
        newgens = (I - Y)*gens*normdivisor
        #print newgens
        contribution = sqrt(det(newgens.transpose()*newgens)/oldvol)
    return contribution


    
class PartitionLattice:
    def __init__(self, groundset):
        self.groundset = groundset
        self.zero = [[i + 1] for i in range(groundset)]
        self.one = [[i + 1 for i in range(groundset)]]
        self.chains = chains(groundset)
        self.group = SymmetricGroup(groundset)
        
    

#list comp for group elt H[0] acting on chain C
#[[[H[0](C[j][k][i]) for i in range(len(C[j][k]))]for k in range(len(C[j]))]for j in range(len(C))]            
            
#list comp for canonical chain rep hack chain x3
#[[[list(set(x3[i][j])) for j in range(len(x3[i]))]for i in range(len(x3))]]
