#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 17:46:58 2021

@author: alberto
"""

import numpy as np
import scipy.sparse as sslin
import scipy.sparse.linalg as scip_sparse_linalg

class linear_operators_class:
    
    def __init__(self,jet,rc,zc,direction):
        
        # Linear operators for the forward solves
        self.D = assemble_divergence(jet)
        self.G = assemble_gradient(jet)
        self.L = assemble_laplacian(jet)
        self.Lp = augment_pressure_laplacian(jet,self.D.dot(self.G))
        self.luLp = scip_sparse_linalg.splu(self.Lp)
        self.BC_op = assemble_boundary_condition_operator(jet)
        
        # Linear operators for the adjoint solves
        self.DT = self.D.T
        self.GT = self.G.T
        self.LT = self.L.T
        self.luLpT = scip_sparse_linalg.splu((self.Lp.T).tocsc())
        self.BC_op_T = self.BC_op.transpose()
        
        # Weights and operators to add/remove the ghost points 
        self.W, self.Winv, self.Wsqrt, self.Wsqrtinv, self.Wp, self.Wp_inv, self.Wpsqrt, self.Wpsqrt_inv = assemble_weights(jet)
        self.Mrem = remove_boundary_points_operator(jet)
        self.Mapp = self.Mrem.transpose()
        
        # Input operator
        self.B = assemble_input_operator(jet,rc,zc,direction)
        
        
# ---------------------------------------------------------------------------
# ----------- This is the B matrix for an external input --------------------
# ---------------------------------------------------------------------------        

def assemble_input_operator(jet,rc,zc,direction):
    
    B = np.zeros(jet.szu + jet.szv)
    
    if direction == 'axial':
                
        for i in range (1,jet.rowsu-1):
            r = jet.r[i-1]
            for j in range (1,jet.colsu-1):
                k = i*jet.colsu + j 
                z = 0.5*(jet.z[j] + jet.z[j-1])
                
                exponent = (1/jet.theta0)*((r - rc)**2 + (z - zc)**2)
                B[k] = np.exp(-exponent)
                
    if direction == 'radial':
        
        for i in range (1,jet.rowsv-1):
            r = 0.5*(jet.r[i] + jet.r[i-1])
            for j in range (1,jet.colsv-1):
                k = i*jet.colsu + j 
                z = jet.z[j-1]
                
                exponent = (1/jet.theta0)*((r - rc)**2 + (z - zc)**2)
                B[k] = np.exp(-exponent)
        
    
    return B



# ---------------------------------------------------------------------------
# ---------- This operator takes a vector and populates  --------------------
# ---------- the ghost points with the appropriate boundary conditions ------
# ---------------------------------------------------------------------------

def assemble_boundary_condition_operator(jet):
    
    rows = []; cols = []; data = []
    
    # -------------------------------------------------------
    # ---------- Axial velocity boundary conditions ---------
    # -------------------------------------------------------
    for i in range (0,jet.rowsu):
        k_in = i*jet.colsu
        k_out = (i+1)*jet.colsu - 1
        
        if i == jet.rowsu-1:
            rows.extend([k_in,k_out])
            cols.extend([k_in,k_out-1-jet.colsu])

        elif i == 0:
            rows.extend([k_in,k_out])
            cols.extend([k_in,k_out-1+jet.colsu])
        else:
            rows.extend([k_in,k_out])
            cols.extend([k_in,k_out-1])
        
        data.extend([0.0,1.0])
        
    for j in range (1,jet.colsu-1):
        k_bot = j
        k_top = (jet.rowsu-1)*jet.colsu + j
        
        rows.extend([k_bot,k_top])
        cols.extend([k_bot+jet.colsu,k_top-jet.colsu])
        data.extend([1.0,1.0])
        
    for i in range (1,jet.rowsu-1):
        for j in range (1,jet.colsu-1):
            k = i*jet.colsu + j
            
            rows.append(k)
            cols.append(k)
            data.append(1.0)
        
    # -------------------------------------------------------
    # ---------- Radial velocity boundary conditions --------
    # -------------------------------------------------------
    for i in range (0,jet.rowsv):
        k_in = i*jet.colsv + jet.szu
        k_out = (i+1)*jet.colsv - 1 + jet.szu
        
        rows.extend([k_in,k_out])
        cols.extend([k_in+1,k_out-1])
        data.extend([-1.0,1.0])
        
    for j in range (1,jet.colsv-1):
        k_bot = j + jet.szu
        k_top = (jet.rowsv-1)*jet.colsv + j + jet.szu
        
        rows.extend([k_bot,k_top])
        cols.extend([k_bot+jet.colsv,k_top-jet.colsv])
        data.extend([0.0,1.0])
        
    for i in range (1,jet.rowsv-1):
        for j in range (1,jet.colsv-1):
            k = i*jet.colsv + j + jet.szu
            
            rows.append(k)
            cols.append(k)
            data.append(1.0)
        
    M = sslin.csc_matrix((data,(rows,cols)),shape=(jet.szu+jet.szv,jet.szu+jet.szv))
    
    return M



def remove_boundary_points_operator(jet):
    
    rowsu = len(jet.r)
    colsu = len(jet.z) - 1
    
    rowsv = len(jet.r) - 1
    colsv = len(jet.z)
    
    data = []; rows = []; cols = []
    for i in range (rowsu):
        for j in range (colsu):
            k_rem = i*colsu + j
            k = (i+1)*jet.colsu + (j+1)
            
            rows.append(k_rem)
            cols.append(k)
            data.append(1.0)

            
    for i in range (rowsv):
        for j in range (colsv):
            k_rem = i*colsv + j + rowsu*colsu
            k = (i+1)*jet.colsv + (j+1) + jet.szu
            
            rows.append(k_rem)
            cols.append(k)
            data.append(1.0)
            
    M = sslin.csc_matrix((data,(rows,cols)),shape=(rowsu*colsu + rowsv*colsv,jet.szu+jet.szv))
            
    return M




# ------------------------------------------------------------------------
# ---------- The operators below are the linear operators that -----------
# ---------- appear in the Navier-Stokes equation ------------------------
# ------------------------------------------------------------------------

        
def assemble_laplacian(jet):
    
    dr = jet.r[1] - jet.r[0]
    dz = jet.z[1] - jet.z[0]
    
    rows = []; cols = []; data = []
    
    for i in range (1,jet.rowsu-1):
        r = jet.r[i-1]
        for j in range (1,jet.colsu-1):
            k = i*jet.colsu + j
            
            valm1r = 1/(dr*dr*jet.Re) - 1/(2*r*dr*jet.Re)
            valm1z = 1/(dz*dz*jet.Re)
            val    = -2/(dz*dz*jet.Re) - 2/(dr*dr*jet.Re)
            valp1z = 1/(dz*dz*jet.Re)
            valp1r = 1/(dr*dr*jet.Re) + 1/(2*r*dr*jet.Re)
            
            rows.extend([k,k,k,k,k])
            cols.extend([k-jet.colsu,k-1,k,k+1,k+jet.colsu])
            data.extend([valm1r,valm1z,val,valp1z,valp1r])
            
    
    for i in range (1,jet.rowsv-1):
        r = 0.5*(jet.r[i] + jet.r[i-1])
        for j in range (1,jet.colsv-1):
            k = i*jet.colsv + j + jet.szu
            
            valm1r = 1/(dr*dr*jet.Re) - 1/(2*r*dr*jet.Re)
            valm1z = 1/(dz*dz*jet.Re)
            val    = -2/(dz*dz*jet.Re) - 2/(dr*dr*jet.Re) - 1/(r*r*jet.Re)
            valp1z = 1/(dz*dz*jet.Re)
            valp1r = 1/(dr*dr*jet.Re) + 1/(2*r*dr*jet.Re)
            
            rows.extend([k,k,k,k,k])
            cols.extend([k-jet.colsv,k-1,k,k+1,k+jet.colsv])
            data.extend([valm1r,valm1z,val,valp1z,valp1r])
    
    L = sslin.csc_matrix((data,(rows,cols)),shape=(jet.szu+jet.szv,jet.szu+jet.szv))
    
    return L


def assemble_divergence(jet):
    
    dr = jet.r[1] - jet.r[0]
    dz = jet.z[1] - jet.z[0]
    
    
    rows = []; cols = []; data = []
    
    for i in range (0,jet.rowsp):
        r = jet.r[i]
        for j in range (0,jet.colsp):
            
            kp = i*jet.colsp + j
            ku = (i+1)*jet.colsu + (j+1)
            kv = (i+1)*jet.colsv + (j+1) + jet.szu
            
            rows.extend([kp,kp,kp,kp])
            cols.extend([ku,ku-1,kv,kv-jet.colsv])
            data.extend([1/dz,-1/dz,1/dr+0.5/r,-1/dr+0.5/r])
            
    D = sslin.csc_matrix((data,(rows,cols)),shape=(jet.szp,jet.szu+jet.szv))
    
    return D


def assemble_gradient(jet):
    
    dr = jet.r[1] - jet.r[0]
    dz = jet.z[1] - jet.z[0]
    
    rows = []; cols = []; data = []
    
    for i in range (1,jet.rowsu-1):
        for j in range (1,jet.colsu-1):
            
            ku = i*jet.colsu + j
            kp = (i-1)*jet.colsp + (j-1)
            
            rows.extend([ku,ku])
            cols.extend([kp,kp+1])
            data.extend([-1/dz,1/dz])
    
    for i in range (1,jet.rowsv-1):
        for j in range (1,jet.colsv-1):
            
            ku = i*jet.colsv + j + jet.szu
            kp = (i-1)*jet.colsp + (j-1)
            
            rows.extend([ku,ku])
            cols.extend([kp,kp+jet.colsp])
            data.extend([-1/dr,1/dr])
         
    G = sslin.csc_matrix((data,(rows,cols)),shape=(jet.szu+jet.szv,jet.szp))
    
    return G


def augment_pressure_laplacian(jet,DG):
        
    rows = []; cols = []; data = []
    
    valr = 1/np.sqrt(jet.szp-1)
    valc = 0
    for i in range (0,jet.rowsp):
        for j in range (0,jet.colsp):
            valc = valc + (2*jet.r[i])**2
    
    valc = 1/np.sqrt(valc)
    
    for k in range (0,jet.szp-1):
        rows.extend([jet.szp-1,k])
        cols.extend([k,jet.szp-1])
        data.extend([valr,2*jet.r[k//jet.colsp]*valc])
        
    M = sslin.csc_matrix((data,(rows,cols)),shape=(jet.szp,jet.szp))
    
    return M + DG
    
    
    
def assemble_weights(jet):
    
    rows = []; cols = []; data = []; datainv = []
    datasqrt = []; datasqrtinv = []
    
    # ----------------------------------------------------------------
    # ---------- Weight operators for the velocity variables ---------
    # ----------------------------------------------------------------
    
    for i in range (1,jet.rowsu-1):
        r = jet.r[i-1]
        
        for j in range (1,jet.colsu-1):
            k = i*jet.colsu + j
            
            rows.append(k)
            cols.append(k)
            data.append(r)
            datainv.append(1/r)
            datasqrt.append(np.sqrt(r))
            datasqrtinv.append(1/np.sqrt(r))
            
    for i in range (1,jet.rowsv-1):
        r = 0.5*(jet.r[i] + jet.r[i-1])
 
        for j in range (1,jet.colsv-1):
            k = i*jet.colsv + j + jet.szu
            
            rows.append(k)
            cols.append(k)
            data.append(r)
            datainv.append(1/r)
            datasqrt.append(np.sqrt(r))
            datasqrtinv.append(1/np.sqrt(r))
            
    
    W = sslin.csc_matrix((data,(rows,cols)),shape=(jet.szu+jet.szv,jet.szu+jet.szv))   
    Winv = sslin.csc_matrix((datainv,(rows,cols)),shape=(jet.szu+jet.szv,jet.szu+jet.szv))    
    Wsqrt = sslin.csc_matrix((datasqrt,(rows,cols)),shape=(jet.szu+jet.szv,jet.szu+jet.szv))   
    Wsqrtinv = sslin.csc_matrix((datasqrtinv,(rows,cols)),shape=(jet.szu+jet.szv,jet.szu+jet.szv))    
    
    
    rows = []; cols = []; data = []; datainv = []
    datasqrt = []; datasqrtinv = []
    
    # ----------------------------------------------------------------
    # ---------- Weight operators for the pressure variable ----------
    # ----------------------------------------------------------------
    
    for i in range (0,jet.rowsp):
        r = jet.r[i]
        for j in range (0,jet.colsp):
            k = i*jet.colsp + j
            
            rows.append(k)
            cols.append(k)
            data.append(r)
            datainv.append(1/r)
            datasqrt.append(np.sqrt(r))
            datasqrtinv.append(1/np.sqrt(r))

            
    
    Wp = sslin.csc_matrix((data,(rows,cols)),shape=(jet.szp,jet.szp))   
    Wp_inv = sslin.csc_matrix((datainv,(rows,cols)),shape=(jet.szp,jet.szp))     
    Wpsqrt = sslin.csc_matrix((datasqrt,(rows,cols)),shape=(jet.szp,jet.szp))   
    Wpsqrt_inv = sslin.csc_matrix((datasqrtinv,(rows,cols)),shape=(jet.szp,jet.szp)) 
            

    return W, Winv, Wsqrt, Wsqrtinv, Wp, Wp_inv, Wpsqrt, Wpsqrt_inv
    
    
    
    
    
    
    
    
    
    
    
    


