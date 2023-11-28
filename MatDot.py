from sage.all import *
import matplotlib.pyplot as plt
import random

#######################################################
#######################################################

''' #### This commented out section demonstrates a particular implementation of the MatDot coded matrix multiplication. 
q=11^3
N=8
K=4
k.<x>=GF(q) # Fixes a Finite Field of size q
field_elements=[i for i in k] #if i not in [0]]  # enumeration of non-zero field elements
eval=random.sample(field_elements,N)   # list of randomly picked evaluation points

###################################################
## This section generates the source matrices, encodes them at the source, and uploads the encoded shares to the computing cluster
###################################################
dim1=[15,60]
dim2=[60,15]
A=Source([dim1[0],dim1[1]])
A.gen_data()
A.matdot_lencode(N,K)
B=Source([dim2[0],dim2[1]])
B.gen_data()
B.matdot_rencode(N,K)

comp_cluster=Cluster(N)
A.uploadtonodes(comp_cluster,K)
B.uploadtonodes(comp_cluster,K)


######################################################
##### This section illustrates the computation of matrix product A*B.
######################################################

servers_multiply()

############ The user interpolates on the received matrices to obtain a polynomial whose (K-1)-th term is the desired matric product  #############

fast_servers=random.sample(range(N),2*K-1)   # the servers that return the values first, i.e., non-stragglers, simulated by randomly sampling from the N servers.

C = reconstruct(fast_servers)

verify = C - A.mat*B.mat
#print(verify)  # verify must be an all zero matrix if the result is correct.
'''


class WorkerNodes(object):
    def __init__(self, P):
        self.worker_nodes = []
        for i in range(P):  # P: number of worker nodes in the cluster
            self.worker_nodes.append(WorkerNode())  # instantiating a list of worker nodes

    def get_worker_node(self, worker_idx):
        return self.worker_nodes[worker_idx]

    def compute(self, P):
        '''
        Let all workers(indexed from 0 to P-1) do computation.
        '''
        for i in range(P):
            worker_node = comp_cluster.get_worker_node(i)
            worker_node.multiply()


########################################################
####################################################
#####


class WorkerNode(object):
    def __init__(self):
        # Each element of info_of_sub_matrix_X is [sub_matrix_X, P, m]
        self.info_of_sub_matrix_A = []
        self.info_of_sub_matrix_B = []
        self.info_of_sub_matrix_C = []  # to be computed

    def get_sub_matrix_A(self):
        return self.info_of_sub_matrix_A[0]

    def get_sub_matrix_B(self):
        return self.info_of_sub_matrix_B[0]

    def get_sub_matrix_C(self):
        return self.info_of_sub_matrix_C[0]

    def set_info_of_sub_matrix_A(self, info_of_sub_matrix_A):
        self.info_of_sub_matrix_A = info_of_sub_matrix_A

    def set_info_of_sub_matrix_B(self, info_of_sub_matrix_B):
        self.info_of_sub_matrix_B = info_of_sub_matrix_B

    def get_P(self):
        return self.info_of_sub_matrix_A[1]

    def get_m(self):
        return self.info_of_sub_matrix_A[2]

    #############################
    def multiply(self):  # ind1 and ind2 are the indexes of the matrices that are multiplied
        sub_matrix_A = self.get_sub_matrix_A()
        sub_matrix_B = self.get_sub_matrix_B()

        sub_matrix_C = sub_matrix_A * sub_matrix_B
        P = self.get_P()
        m = self.get_m()

        self.info_of_sub_matrix_C = [sub_matrix_C, P, m]


#############################
# def add(self,ind1,ind2):
#     # ind1 and ind2 are the indexes of the matrices that are added
#     tmp = [self.mats[ind1][0] + self.mats[ind2][0],self.mats[ind1][1],self.mats[ind1][2]]
#     self.mats.append( tmp )


########################################################
########################################################


class MasterNode(object):
    def __init__(self, finiteField, row_num, col_num):
        '''
        Create a null matrix with given size.
        '''
        self.finiteField = finiteField
        self.row_num = row_num
        self.col_num = col_num

        self.matrix = zero_matrix(finiteField, self.row_num, self.col_num)  # matrix of all zeroes

        # print(f"matrix: {self.mat}")
        # self.encoded_mat=matrix(k,[])

    #############################
    # Function generates a random matrix as the source data
    #############################
    def fill_matrix_with_random_data(self):
        '''
        Fill the matrix with random data.
        '''
        self.matrix = random_matrix(finiteField, self.row_num, self.col_num)

    #############################
    # Function left-encodes the matrix stored in the source instance
    #############################
    def encode_A_with_matdot(self, P, m):  # encodes the left-matrix (i.e. matrix A)
        '''
        Encode the matrix A.
        :param P:
        :param m:
        :return:
        '''
        # A_i:  dimensional sub-matrix.
        # From the paper, since A is N * N, each sub-matrix is N * (N/m).
        # But since A here is N * (N*m), each sub-matrix is N * N here now. Why??
        row_num_of_sub_matrix = self.row_num
        col_num_of_sub_matrix = self.col_num / m

        # We split A horizontally into m sub-matrices.
        # Since A is N * (N*m) and each sub-matrix is N*N, the start col index of each sub-matrix is {0, N, 2N, ..., (m-1)N}
        start_col_indices_of_each_sub_matrix = [(col_num_of_sub_matrix * i) for i in range(1, m)]
        # Now we have sub-matrices A_0, A_1, ..., A_{m-1}, each is N * N.
        self.matrix.subdivide([], start_col_indices_of_each_sub_matrix)

        # We need to deliver sub-matrices(each N*N) to `P` worker nodes, thus we construct a N * (N*p) encoded matrix.
        self.encoded_mat = zero_matrix(self.finiteField, self.row_num, col_num_of_sub_matrix * P)

        # Since the encoded matrix is N * (N*P) and each sub-matrix is N*N, the start col index of each sub-matrix is {0, N, 2N, ..., (P-1)N}
        start_col_indices_of_each_sub_matrix = [col_num_of_sub_matrix * i for i in range(1, P)]  #
        # Now we have sub-encoded-matrices A_0, A_1, ..., A_{P-1}, each is N * N.
        self.encoded_mat.subdivide([], start_col_indices_of_each_sub_matrix)

        # Can't understand.
        for i in range(P):
            sub_matrix = zero_matrix(finiteField, self.row_num, self.col_num / m)  # sub-matrix with size:
            for j in range(m):
                sub_matrix = sub_matrix + self.matrix.subdivision(0, j) * eval[i] ** j

            if i == 0:
                temp_mat = sub_matrix
            else:
                temp_mat = block_matrix([[temp_mat, sub_matrix]])
        self.encoded_mat = self.encoded_mat + temp_mat

        start_col_indices_of_each_sub_matrix = [col_num_of_sub_matrix * i for i in range(1, P)]
        self.encoded_mat.subdivide([], start_col_indices_of_each_sub_matrix)

    #############################
    def encode_B_with_matdot(self, P, m):  # encodes the right-matrix (i.e., matrix B)
        '''
        Encode the matrix B.
        '''
        # B_i: N/m × N dimensional sub-matrix.
        s_key = [self.row_num / m, self.col_num]
        sub_ind = [i * s_key[0] for i in range(1, m)]
        self.matrix.subdivide(sub_ind, [])
        self.encoded_mat = zero_matrix(finiteField, s_key[0] * P, s_key[1])
        sub_ind = [s_key[0] * i for i in range(1, P)]
        self.encoded_mat.subdivide(sub_ind, [])

        for i in range(P):
            submat = zero_matrix(finiteField, self.row_num / m, self.col_num)
            for j in range(m):
                # submat=submat + self.mat.subdivision(j,0)*eval[i]^(K-1-j)
                submat = submat + self.matrix.subdivision(j, 0) * eval[i] ** (m - 1 - j)
            if i == 0:
                temp_mat = submat
            else:
                temp_mat = block_matrix([[temp_mat], [submat]])

        self.encoded_mat = self.encoded_mat + temp_mat
        sub_ind = [s_key[0] * i for i in range(1, P)]
        self.encoded_mat.subdivide(sub_ind, [])

    ###############################
    # Function sends the encoded matrices to the worker nodes
    ###############################
    def upload_to_worker_nodes(self, clstr, P, m):
        for i in range(P):
            worker_node_i = clstr.get_worker_node(i)
            if is_vertically_divided(self.encoded_mat):
                sub_matrix_A = get_the_i_th_vertical_sub_matrix(self.encoded_mat, i)
                worker_node_i.set_info_of_sub_matrix_A(
                    [sub_matrix_A, P, m])  # Worker node i will get A_i, the i-th sub-matrix of A.
            else:
                sub_matrix_B = get_the_i_th_horizontal_sub_matrix(self.encoded_mat, i)
                worker_node_i.set_info_of_sub_matrix_B(
                    [sub_matrix_B, P, m])  # Worker node i will get B_i, the i-th sub-matrix of B.


####################################################################
# The main program code acts as the user:
####################################################################

def is_vertically_divided(divided_matrix):
    '''
    Given a divided matrix, judge if it's divided vertically.
    Note that matrix A is divided vertically, while matrix B is divided horizontally,
    so if this function returns True, then the given matrix is matrix A, else it's matrix B.
    '''
    sub_divisions = divided_matrix.subdivisions()
    return sub_divisions[0] == []


def get_the_i_th_vertical_sub_matrix(divided_matrix, i):
    '''
    Get an immutable copy of the (0,i)th sub-matrix of self, according to a previously set subdivision.
    Since matrix A has been divided vertically to `m` sub-matricesm and each is N * (N/m), this function will return the ith vertical sub-matrix.
    '''
    sub_matrix_A_i = divided_matrix.subdivision(0, i)
    return sub_matrix_A_i


def get_the_i_th_horizontal_sub_matrix(divided_matrix, i):
    '''
    Get an immutable copy of the (i,0)th sub-matrix of self, according to a previously set subdivision.
    Since matrix A has been divided horizontally to `m` sub-matricesm and each is (N/m) * N, this function will return the ith horizontal sub-matrix.
    '''
    sub_matrix_B_i = divided_matrix.subdivision(i, 0)
    return sub_matrix_B_i



def reconstruct(finiteField, comp_cluster, eval, fast_servers, row_num, col_num, m):
    R = PolynomialRing(finiteField, 'y')
    matrix_C = zero_matrix(finiteField, row_num,
                           col_num)  # comp_cluster.nodes[0].mats[2][0].nrows(),comp_cluster.nodes[0].mats[2][0].ncols())

    # What is the `list_points` and `lagrange_polynomial` here?
    worker_node = comp_cluster.get_worker_node(0)
    row_num_of_sub_matrix_C_of_worker_node = worker_node.get_sub_matrix_C().nrows()
    col_num_of_sub_matrix_C_of_worker_node = worker_node.get_sub_matrix_C().ncols()

    for i in range(row_num_of_sub_matrix_C_of_worker_node):
        for j in range(col_num_of_sub_matrix_C_of_worker_node):
            k = 2 * m - 1
            list_points = [(0, 0)] * (k)
            # at the (i,j)-th position of the resultant matrices, list_points[] forms the list of the evaluation points of the received matrices. By interpolating on the points in list_points, we will obtain the polynomial matrix_C(x) at the (i,j)-th position
            for l in range(k):
                list_points[l] = (
                eval[fast_servers[l]], comp_cluster.get_worker_node(fast_servers[l]).get_sub_matrix_C()[i][j])
            f = R.lagrange_polynomial(list_points)
            '''
            ##### The following manner of listing the polynomial coefficients excluded the zero terms, which caused a misalignment in the list of coefficients. This was leading to occasional errors in the output. Hence, the argument 'sparse=False' is set so that the zero coefficients are also listed. #########
            matrix_C[i,j]=f.coefficients()[K-1]
            '''
            matrix_C[i, j] = f.coefficients(sparse=False)[
                m - 1]  # matrix_C is the coefficient of x^{m−1} in the product p_C(x) = p_A(x)p_B(x)

    return matrix_C


def __is_zero_matrix(matrix):
    return all(all(element == 0 for element in row) for row in matrix)


if __name__ == "__main__":
    '''
        We define a Computational System (N,k,P,m) with initial value:
        * N = 15
        * k = 2*m - 1 = 7
        * P = 8
        * m = 4 
    '''
    P = 8  # num of workers
    m = 4  # the memory of each worker in the model, i.e., each worker node can store only upto a $1/m$ fraction of each of the input matrices.
    k = 2 * m - 1  # recovery threshold k = 2m -1
    N = 15

    ## This commented out section demonstrates a particular implementation of the MatDot coded matrix multiplication.
    q = 11 ^ 3
    finiteField = GF(q, 'x')  # Fixes a Finite Field of size q

    field_elements = [i for i in finiteField]  # if i not in [0]]  # enumeration of non-zero field elements
    eval = random.sample(field_elements, P)  # list of randomly picked evaluation points

    # From the paper, A and B should all be N*N matrices. But in this code they're not. IDK why???
    row_num_of_matrix_A = N
    col_num_of_matrix_A = N * m

    row_num_of_matrix_B = N * m
    col_num_of_matrix_B = N

    # Initialize matrix A, B.
    master_node_for_matrix_A = MasterNode(finiteField, row_num_of_matrix_A, col_num_of_matrix_A)
    master_node_for_matrix_B = MasterNode(finiteField, row_num_of_matrix_B, col_num_of_matrix_B)

    # Assign values for  A,B. Since it's just experiment. We fill A, B with random number.
    master_node_for_matrix_A.fill_matrix_with_random_data()
    master_node_for_matrix_B.fill_matrix_with_random_data()

    # Encode A, B so that they can be sent to worker nodes.
    master_node_for_matrix_A.encode_A_with_matdot(P, m)
    master_node_for_matrix_B.encode_B_with_matdot(P, m)

    # Initialize `P` worker nodes.
    comp_cluster = WorkerNodes(P)

    # Send A, B to worker nodes.
    master_node_for_matrix_A.upload_to_worker_nodes(comp_cluster, P, m)
    master_node_for_matrix_B.upload_to_worker_nodes(comp_cluster, P, m)
    # Now each worker node has it's sub-matrix A and sub-matrix B.

    # The worker nodes now compute their tasks.
    comp_cluster.compute(P)

    # The user interpolates on the received matrices to obtain a polynomial whose (K-1)-th term is the desired matric product  #############
    # Randomly choose the indices of `k` unique worker nodes. The indices are in {0,1,...,P-1}.
    # These `k` worker nodes represents the `k` successful, i.e., non-stragglers, worker nodes in a real system.
    indices_of_successful_worker_nodes = random.sample(range(P), k)

    matrix_C_reconstructed = reconstruct(finiteField, comp_cluster, eval, indices_of_successful_worker_nodes,
                                         row_num_of_matrix_A, col_num_of_matrix_B, m)
    matrix_C_directly_computed = master_node_for_matrix_A.matrix * master_node_for_matrix_B.matrix

    verify = matrix_C_reconstructed - matrix_C_directly_computed
    if __is_zero_matrix(verify):  # If the verify matrix is all-zero, then A*B = MatDot(A,B).
        print("Successful! The reconstructed matrix C equals to the result of A * B")

        # print(verify)

