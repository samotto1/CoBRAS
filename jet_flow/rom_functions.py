#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 19:33:55 2022

@author: apadovan
"""

import numpy as np
import nonlinear_operators_no_upwind_logic as nonlin_ops


def enforce_div_free(lops,q):

    p = lops.luLp.solve(lops.D.dot(q))
    q[:] = q - lops.G.dot(p)

    return q


def assemble_rom(jet,lops,Phi,Psi):
    
    # Inputs:       Phi, Psi in x coordinates
    
    # Outputs:      Al (r x r matrix from linear dynamics)
    #               Abl (r x r x r tensor from bilinear term)
    #               Bl (r x m) input matrix
    
    Psi = lops.Mrem.dot(lops.Wsqrt.dot(lops.Mapp.dot(Psi)))
    Phi = lops.BC_op.dot(lops.Wsqrtinv.dot(lops.Mapp.dot(Phi)))
    N, r = Phi.shape
    
    q = np.zeros(N)
    qi = np.zeros(N)
    qj = np.zeros(N)
    qij = np.zeros(N)
    
    # ---- Assemble the linear rom dynamics -------
    Al = np.zeros((r,r))
    for j in range (r):
        q[:] = Phi[:,j]
        q[:] = enforce_div_free(lops,q)
        q[:] = lops.L.dot(q) - nonlin_ops.evaluate_bilinearity(jet.r,jet.z,jet.q_sbf,q) - \
                nonlin_ops.evaluate_bilinearity(jet.r,jet.z,q,jet.q_sbf)
        q[:] = enforce_div_free(lops,q)         # New line to enforce div free
        Al[:,j] = np.dot(Psi.T,lops.Mrem.dot(lops.BC_op.dot(q)))
    
    # ---- Assemble the bilinear rom dynamics -----
    Abl = np.zeros((r,r,r))
    for i in range (r):
        qi[:] = Phi[:,i]
        qi[:] = enforce_div_free(lops,qi)
        for j in range (r):
            qj[:] = Phi[:,j]
            qj[:] = enforce_div_free(lops,qj)
            qij[:] = -nonlin_ops.evaluate_bilinearity(jet.r,jet.z,qi,qj)   
            qij[:] = enforce_div_free(lops,qij)  # New line to enforce div free       
            Abl[:,i,j] = np.dot(Psi.T,lops.Mrem.dot(lops.BC_op.dot(qij)))
    
    # ---- Assemble the volumetric forcing --------    
    Bl = np.dot(Psi.T,lops.Mrem.dot(enforce_div_free(lops,lops.B)))
            
                
    return Al, Abl, Bl

    
def evaluate_f(jet,lops,x,u):
    
    # Convert from x to q coordinates
    q = lops.BC_op.dot(lops.Wsqrtinv.dot(lops.Mapp.dot(x))) + jet.q_sbf
    
    # Evaluate f(q)
    p = lops.luLp.solve(lops.D.dot(q))
    q[:] = q - lops.G.dot(p)
    f_nonlin = -nonlin_ops.evaluate_bilinearity(jet.r,jet.z,q,q)
    f = f_nonlin + lops.L.dot(q) + lops.B*u
    p = lops.luLp.solve(lops.D.dot(f))
    f[:] = f - lops.G.dot(p)
    
    # Convert back to x coordinates
    f_out = lops.Mrem.dot(lops.Wsqrt.dot(lops.BC_op.dot(f)))
    
    return f_out


def evaluate_df(jet,lops,x,u,v):
    
    # Convert from x to q coordinates
    q = lops.BC_op.dot(lops.Wsqrtinv.dot(lops.Mapp.dot(x))) + jet.q_sbf
    v_q = lops.BC_op.dot(lops.Wsqrtinv.dot(lops.Mapp.dot(v)))
    
    # Evaluate df(q,u)*lam
    p = lops.luLp.solve(lops.D.dot(v_q))
    v_q[:] = v_q - lops.G.dot(p)
    p = lops.luLp.solve(lops.D.dot(q))
    q[:] = q - lops.G.dot(p)
    df_bilin = -nonlin_ops.evaluate_bilinearity(jet.r,jet.z,q,v_q) - \
                nonlin_ops.evaluate_bilinearity(jet.r,jet.z,v_q,q)
    df = df_bilin + lops.L.dot(v_q)
    p = lops.luLp.solve(lops.D.dot(df))
    df[:] = df - lops.G.dot(p)
    
    # Convert back to x coordinates
    df_out = lops.Mrem.dot(lops.Wsqrt.dot(lops.BC_op.dot(df)))
    
    return df_out
    
    

def evaluate_df_adj(jet,lops,x,u,v):
    
    # Convert from x to q coordinates
    q = lops.BC_op.dot(lops.Wsqrtinv.dot(lops.Mapp.dot(x))) + jet.q_sbf
    lam = lops.BC_op_T.dot(lops.Wsqrtinv.dot(lops.Mapp.dot(v)))
    
    # Ensure that q is divergence free
    p = lops.luLp.solve(lops.D.dot(q))
    q[:] = q - lops.G.dot(p)
    
    # # Evaluate Winv*[df(q,u)]^T*W*lam
    lam = lops.BC_op_T.dot(lops.W.dot(lam))
    p = lops.luLpT.solve(lops.GT.dot(lam))
    lam[:] = lam - lops.DT.dot(p)
    dfT_bilin = -nonlin_ops.evaluate_adjoint_advection(jet.r,jet.z,q,lam) - \
                nonlin_ops.evaluate_adjoint_static(jet.r,jet.z,lam,q)
    dfT = dfT_bilin + lops.LT.dot(lam)
    p = lops.luLpT.solve(lops.GT.dot(dfT))
    dfT[:] = dfT - lops.DT.dot(p)
    dfT[:] = lops.Winv.dot(lops.BC_op_T.dot(dfT))
    
    # Convert back to x coordinates (i.e., compute [df(x,u)]^T*v)
    dfT_out = lops.Mrem.dot(lops.Wsqrt.dot(dfT))
    
    # dfT_out = dfT
    
    return dfT_out

    
def compute_div_free_B_matrix(lops):
    
    p = lops.luLp.solve(lops.D.dot(lops.B))
    Bdf = lops.Mrem.dot(lops.B - lops.G.dot(p))
    
    return Bdf
        
    
            
            
        
        
        
            
            
            
            
