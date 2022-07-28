#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 19:16:36 2021

@author: alberto
"""

import numpy as np
import numba 


@numba.njit("f8[:](f8[:],f8[:],f8[:],f8[:])",cache=True) 
def evaluate_bilinearity(r,z,q1,q2):
    
    dr = r[1] - r[0]
    dz = z[1] - z[0]
    
    rowsu = len(r) + 2
    colsu = len(z) + 1
    szu = rowsu*colsu
    
    rowsv = len(r) + 1
    colsv = len(z) + 2
    szv = rowsv*colsv
    
    qnl = np.zeros(szu+szv)
    # -----------------------------------------------
    # ---------- Axial momentum ---------------------
    # -----------------------------------------------
    
    for i in range (1,rowsu-1):
        for j in range (1,colsu-1):
            
            ku = i*colsu + j
            kv = i*colsv + j + szu
            
            u = q1[ku]
            v = 0.25*(q1[kv] + q1[kv+1] + q1[kv-colsv] + q1[kv-colsv+1])
            
            up = u; um = 0
            vp = v; vm = 0
            
            
            # -------------------------------------------------
            # -------- z derivatives --------------------------
            # -------------------------------------------------
            if j > 1 and j < colsu-2:
            
                qnl[ku] +=  up*(2*q2[ku+1]   + 3*q2[ku]  - 6*q2[ku-1]  + q2[ku-2])/(6*dz) + \
                            um*(-2*q2[ku-1]  - 3*q2[ku]  + 6*q2[ku+1]  - q2[ku+2])/(6*dz)
                            
            elif j == 1:
                
                qnl[ku] +=  up*(q2[ku] - q2[ku-1])/dz + \
                            um*(-2*q2[ku-1]  - 3*q2[ku]  + 6*q2[ku+1]  - q2[ku+2])/(6*dz)
                            
            elif j == colsu-2:
                
                qnl[ku] +=  up*(2*q2[ku+1]   + 3*q2[ku]  - 6*q2[ku-1]  + q2[ku-2])/(6*dz) + \
                            um*(q2[ku+1] - q2[ku])/dz
            
            
            # -------------------------------------------------
            # -------- r derivatives --------------------------
            # -------------------------------------------------
            if i > 1 and i < rowsu-2:
                
                qnl[ku] +=  vp*(2*q2[ku+colsu]   + 3*q2[ku]  - 6*q2[ku-colsu]   + q2[ku-2*colsu])/(6*dr) + \
                            vm*(-2*q2[ku-colsu]  - 3*q2[ku]  + 6*q2[ku+colsu]   - q2[ku+2*colsu])/(6*dr) 
                            
            elif i == 1:
                
                qnl[ku] +=  vp*(q2[ku] - q2[ku-colsu])/dr + \
                            vm*(-2*q2[ku-colsu]  - 3*q2[ku]  + 6*q2[ku+colsu]   - q2[ku+2*colsu])/(6*dr) 
                            
            elif i == rowsu-2:
                
                qnl[ku] +=  vp*(2*q2[ku+colsu]   + 3*q2[ku]  - 6*q2[ku-colsu]   + q2[ku-2*colsu])/(6*dr) + \
                            vm*(q2[ku+colsu] - q2[ku])/dr
                            
    
    # -----------------------------------------------
    # ---------- Radial momentum --------------------
    # -----------------------------------------------
                            
    for i in range (1,rowsv-1):
        for j in range (1,colsv-1):
            
            ku = i*colsu + j
            kv = i*colsv + j + szu
            
            u = 0.25*(q1[ku] + q1[ku+colsu] + q1[ku-1] + q1[ku-1+colsu])
            v = q1[kv]
            
            up = u; um = 0
            vp = v; vm = 0
            
            
            # -------------------------------------------------
            # -------- z derivatives --------------------------
            # -------------------------------------------------
            
            if j > 1 and j < colsv-2:
                
                qnl[kv] +=  up*(2*q2[kv+1]   + 3*q2[kv]  - 6*q2[kv-1]  + q2[kv-2])/(6*dz) + \
                            um*(-2*q2[kv-1]  - 3*q2[kv]  + 6*q2[kv+1]  - q2[kv+2])/(6*dz)
                                                        
            elif j == 1:
                
                qnl[kv] +=  up*(q2[kv] - q2[kv-1])/dz + \
                            um*(-2*q2[kv-1]  - 3*q2[kv]  + 6*q2[kv+1]  - q2[kv+2])/(6*dz)
                            
            elif j == colsv-2:
                
                qnl[kv] +=  up*(2*q2[kv+1]   + 3*q2[kv]  - 6*q2[kv-1]  + q2[kv-2])/(6*dz) + \
                            um*(q2[kv+1] - q2[kv])/dz
                            
            
            # -------------------------------------------------
            # -------- r derivatives --------------------------
            # -------------------------------------------------
            
            if i > 1 and i < rowsv-2:
                
                qnl[kv] +=  vp*(2*q2[kv+colsv]   + 3*q2[kv]  - 6*q2[kv-colsv]   + q2[kv-2*colsv])/(6*dr) + \
                            vm*(-2*q2[kv-colsv]  - 3*q2[kv]  + 6*q2[kv+colsv]   - q2[kv+2*colsv])/(6*dr) 
                            
            elif i == 1:
                
                qnl[kv] +=  vp*(q2[kv] - q2[kv-colsv])/dr + \
                            vm*(-2*q2[kv-colsv]  - 3*q2[kv]  + 6*q2[kv+colsv]   - q2[kv+2*colsv])/(6*dr) 
                            
            elif i == rowsv-2:
                
                qnl[kv] +=  vp*(2*q2[kv+colsv]   + 3*q2[kv]  - 6*q2[kv-colsv]   + q2[kv-2*colsv])/(6*dr) + \
                            vm*(q2[kv+colsv] - q2[kv])/dr
            
    return qnl


@numba.njit("f8[:](f8[:],f8[:],f8[:],f8[:])",cache=True) 
def evaluate_adjoint_advection(r,z,Q,q):
    
    # This function evaluates the adjoint of the linear operator 
    # L(Q)[q] = Q \cdot \nabla q
    
    dr = r[1] - r[0]
    dz = z[1] - z[0]
    
    rowsu = len(r) + 2
    colsu = len(z) + 1
    szu = rowsu*colsu
    
    rowsv = len(r) + 1
    colsv = len(z) + 2
    szv = rowsv*colsv
    
    qnl = np.zeros(szu+szv)
    # -----------------------------------------------
    # ---------- Axial momentum ---------------------
    # -----------------------------------------------
    
    for i in range (1,rowsu-1):
        for j in range (1,colsu-1):
            
            ku = i*colsu + j
            kv = i*colsv + j + szu
            
            u = Q[ku]
            v = 0.25*(Q[kv] + Q[kv+1] + Q[kv-colsv] + Q[kv-colsv+1])
            
            up = u; um = 0
            vp = v; vm = 0
            
            # -------------------------------------------------
            # -------- z derivatives --------------------------
            # -------------------------------------------------
            if j > 1 and j < colsu-2:
            
                # Reference indices from the forward simulation
                # qnl[ku] +=  up*(2*q[ku+1]   + 3*q[ku]  - 6*q[ku-1]  + q[ku-2])/(6*dz) + \
                #             um*(-2*q[ku-1]  - 3*q[ku]  + 6*q[ku+1]  - q[ku+2])/(6*dz) + \
                #             q[ku]*(Q[ku+1] - Q[ku-1])/(2*dz)
                            
                qnl[ku] += (up*3*q[ku] - um*3*q[ku])/(6*dz) 
                qnl[ku+1] += (up*2*q[ku] + um*6*q[ku])/(6*dz)
                qnl[ku-1] += (-um*2*q[ku] - up*6*q[ku])/(6*dz)
                qnl[ku+2] += -um*q[ku]/(6*dz)
                qnl[ku-2] += up*q[ku]/(6*dz)
                
            elif j == 1:
                
                # Reference indices from the forward simulation
                # qnl[ku] +=  up*(q[ku] - q[ku-1])/dz + \
                #             um*(-2*q[ku-1]  - 3*q[ku]  + 6*q[ku+1]  - q[ku+2])/(6*dz) + \
                #             q[ku]*(Q[ku+1] - Q[ku-1])/(2*dz)
                            
                qnl[ku] += up*q[ku]/dz - 3*um*q[ku]/(6*dz) 
                qnl[ku-1] += -up*q[ku]/dz - 2*um*q[ku]/(6*dz)
                qnl[ku+1] += 6*um*q[ku]/(6*dz)
                qnl[ku+2] += -um*q[ku]/(6*dz)
                            
            elif j == colsu-2:
                
                # Reference indices for the forward simulation
                # qnl[ku] +=  up*(2*q[ku+1]   + 3*q[ku]  - 6*q[ku-1]  + q[ku-2])/(6*dz) + \
                #             um*(q[ku+1] - q[ku])/dz + q[ku]*(Q[ku+1] - Q[ku-1])/(2*dz)
                            
                qnl[ku] += 3*up*q[ku]/(6*dz) - um*q[ku]/dz 
                qnl[ku-1] += -up*6*q[ku]/(6*dz)
                qnl[ku+1] += um*q[ku]/dz + 2*up*q[ku]/(6*dz)
                qnl[ku-2] += up*q[ku]/(6*dz)
            
            
            # -------------------------------------------------
            # -------- r derivatives --------------------------
            # -------------------------------------------------
            if i > 1 and i < rowsu-2:
                
                # Reference indices for forward simulation
                # qnl[ku] +=  vp*(2*q[ku+colsu]   + 3*q[ku]  - 6*q[ku-colsu]   + q[ku-2*colsu])/(6*dr) + \
                #             vm*(-2*q[ku-colsu]  - 3*q[ku]  + 6*q[ku+colsu]   - q[ku+2*colsu])/(6*dr) + \
                #             0.25*(q[kv] + q[kv+1] + q[kv-colsv] + q[kv-colsv+1])*(Q[ku+colsu] - Q[ku-colsu])/(2*dr)
                            
                qnl[ku] += (3*vp*q[ku] - 3*vm*q[ku])/(6*dr)
                qnl[ku+colsu] += (2*vp*q[ku] + 6*vm*q[ku])/(6*dr)
                qnl[ku-colsu] += (-2*vm*q[ku] - 6*vp*q[ku])/(6*dr)
                qnl[ku+2*colsu] += -vm*q[ku]/(6*dr)
                qnl[ku-2*colsu] += vp*q[ku]/(6*dr)
                
                
            elif i == 1:
                
                # Reference indices for forward simulation
                # qnl[ku] +=  vp*(q[ku] - q[ku-colsu])/dr + \
                #             vm*(-2*q[ku-colsu]  - 3*q[ku]  + 6*q[ku+colsu]   - q[ku+2*colsu])/(6*dr) + \
                #             0.25*(q[kv] + q[kv+1] + q[kv-colsv] + q[kv-colsv+1])*(Q[ku+colsu] - Q[ku-colsu])/(2*dr)
                            
                qnl[ku] += vp*q[ku]/dr - 3*vm*q[ku]/(6*dr)
                qnl[ku-colsu] += -vp*q[ku]/dr - 2*vm*q[ku]/(6*dr)
                qnl[ku+colsu] += 6*vm*q[ku]/(6*dr)
                qnl[ku+2*colsu] += -vm*q[ku]/(6*dr)
                            
            elif i == rowsu-2:
                
                # Reference indices for forward simulation
                # qnl[ku] +=  vp*(2*q[ku+colsu]   + 3*q[ku]  - 6*q[ku-colsu]   + q[ku-2*colsu])/(6*dr) + \
                #             vm*(q[ku+colsu] - q[ku])/dr +\
                #             0.25*(q[kv] + q[kv+1] + q[kv-colsv] + q[kv-colsv+1])*(Q[ku+colsu] - Q[ku-colsu])/(2*dr)
                            
                qnl[ku] += vp*3*q[ku]/(6*dr) - vm*q[ku]/dr
                qnl[ku+colsu] += 2*vp*q[ku]/(6*dr) + vm*q[ku]/dr
                qnl[ku-colsu] += -6*vp*q[ku]/(6*dr)
                qnl[ku-2*colsu] += vp*q[ku]/(6*dr)
                
                            
    
    # -----------------------------------------------
    # ---------- Radial momentum --------------------
    # -----------------------------------------------
                            
    for i in range (1,rowsv-1):
        for j in range (1,colsv-1):
            
            ku = i*colsu + j
            kv = i*colsv + j + szu
            
            u = 0.25*(Q[ku] + Q[ku+colsu] + Q[ku-1] + Q[ku-1+colsu])
            v = Q[kv]
            
            up = u; um = 0
            vp = v; vm = 0
            
            # -------------------------------------------------
            # -------- z derivatives --------------------------
            # -------------------------------------------------
            
            if j > 1 and j < colsv-2:
                
                # qnl[kv] +=  up*(2*q[kv+1]   + 3*q[kv]  - 6*q[kv-1]  + q[kv-2])/(6*dz) + \
                #             um*(-2*q[kv-1]  - 3*q[kv]  + 6*q[kv+1]  - q[kv+2])/(6*dz) + \
                #             0.25*(q[ku] + q[ku+colsu] + q[ku-1] + q[ku-1+colsu])*(Q[kv+1] - Q[kv-1])/(2*dz)
                            
                qnl[kv] += (up*3*q[kv] - um*3*q[kv])/(6*dz)
                qnl[kv+1] += (up*2*q[kv] + um*6*q[kv])/(6*dz)
                qnl[kv-1] += (-um*2*q[kv] - 6*up*q[kv])/(6*dz)
                qnl[kv+2] += -um*q[kv]/(6*dz)
                qnl[kv-2] += up*q[kv]/(6*dz)
                                                        
            elif j == 1:
                
                # qnl[kv] +=  up*(q[kv] - q[kv-1])/dz + \
                #             um*(-2*q[kv-1]  - 3*q[kv]  + 6*q[kv+1]  - q[kv+2])/(6*dz) + \
                #             0.25*(q[ku] + q[ku+colsu] + q[ku-1] + q[ku-1+colsu])*(Q[kv+1] - Q[kv-1])/(2*dz)
                
                qnl[kv] += up*q[kv]/dz - um*3*q[kv]/(6*dz)
                qnl[kv+1] += um*6*q[kv]/(6*dz)
                qnl[kv-1] += -um*2*q[kv]/(6*dz) - up*q[kv]/dz
                qnl[kv+2] += -um*q[kv]/(6*dz)
                
                            
            elif j == colsv-2:
                
                # qnl[kv] +=  up*(2*q[kv+1]   + 3*q[kv]  - 6*q[kv-1]  + q[kv-2])/(6*dz) + \
                #             um*(q[kv+1] - q[kv])/dz + \
                #             0.25*(q[ku] + q[ku+colsu] + q[ku-1] + q[ku-1+colsu])*(Q[kv+1] - Q[kv-1])/(2*dz)
                            
                qnl[kv] += up*3*q[kv]/(6*dz) - um*q[kv]/dz
                qnl[kv+1] += up*2*q[kv]/(6*dz) + um*q[kv]/dz
                qnl[kv-1] += - 6*up*q[kv]/(6*dz)
                qnl[kv-2] += up*q[kv]/(6*dz)
                            
            
            # -------------------------------------------------
            # -------- r derivatives --------------------------
            # -------------------------------------------------
            
            if i > 1 and i < rowsv-2:
                
                # qnl[kv] +=  vp*(2*q[kv+colsv]   + 3*q[kv]  - 6*q[kv-colsv]   + q[kv-2*colsv])/(6*dr) + \
                #             vm*(-2*q[kv-colsv]  - 3*q[kv]  + 6*q[kv+colsv]   - q[kv+2*colsv])/(6*dr) + \
                #             q[kv]*(Q[kv+colsv] - Q[kv-colsv])/(2*dr)
                            
                qnl[kv] += (3*vp*q[kv] - 3*vm*q[kv])/(6*dr)
                qnl[kv+colsv] += (vp*2*q[kv] + 6*vm*q[kv])/(6*dr)
                qnl[kv-colsv] += (-2*vm*q[kv] - 6*vp*q[kv])/(6*dr)
                qnl[kv+2*colsv] += -vm*q[kv]/(6*dr)
                qnl[kv-2*colsv] += vp*q[kv]/(6*dr)
                
                            
            elif i == 1:
                
                # qnl[kv] +=  vp*(q[kv] - q[kv-colsv])/dr + \
                #             vm*(-2*q[kv-colsv]  - 3*q[kv]  + 6*q[kv+colsv] - q[kv+2*colsv])/(6*dr) + \
                #             q[kv]*(Q[kv+colsv] - Q[kv-colsv])/(2*dr)
                            
                qnl[kv] += vp*q[kv]/dr - 3*vm*q[kv]/(6*dr) 
                qnl[kv+colsv] += 6*vm*q[kv]/(6*dr)
                qnl[kv-colsv] += -2*vm*q[kv]/(6*dr) - vp*q[kv]/dr
                qnl[kv+2*colsv] += -vm*q[kv]/(6*dr)
                            
            elif i == rowsv-2:
                
                # qnl[kv] +=  vp*(2*q[kv+colsv]   + 3*q[kv]  - 6*q[kv-colsv]   + q[kv-2*colsv])/(6*dr) + \
                #             vm*(q[kv+colsv] - q[kv])/dr + q[kv]*(Q[kv+colsv] - Q[kv-colsv])/(2*dr)
                            
                qnl[kv] += 3*vp*q[kv]/(6*dr) - vm*q[kv]/dr 
                qnl[kv+colsv] += vp*2*q[kv]/(6*dr) + vm*q[kv]/dr
                qnl[kv-colsv] += -6*vp*q[kv]/(6*dr)
                qnl[kv-2*colsv] += vp*q[kv]/(6*dr)
            
            
    return qnl


@numba.njit("f8[:](f8[:],f8[:],f8[:],f8[:])",cache=True) 
def evaluate_adjoint_static(r,z,q1,q2):
    
    # This function evaluates the adjoint of the linear operator 
    # L(Q)[q] = q \cdot \nabla Q
    
    dr = r[1] - r[0]
    dz = z[1] - z[0]
    
    rowsu = len(r) + 2
    colsu = len(z) + 1
    szu = rowsu*colsu
    
    rowsv = len(r) + 1
    colsv = len(z) + 2
    szv = rowsv*colsv
    
    qnl = np.zeros(szu+szv)
    # -----------------------------------------------
    # ---------- Axial momentum ---------------------
    # -----------------------------------------------
    
    for i in range (1,rowsu-1):
        for j in range (1,colsu-1):
            
            ku = i*colsu + j
            kv = i*colsv + j + szu
            
            u = q1[ku]
            v = 0.25*(q1[kv] + q1[kv+1] + q1[kv-colsv] + q1[kv-colsv+1])
            
            up = u; um = 0
            vp = v; vps = 1; vm = 0; vms = 0
            
            # -------------------------------------------------
            # -------- z derivatives --------------------------
            # -------------------------------------------------
            if j > 1 and j < colsu-2:
            
                qnl[ku] +=  up*(2*q2[ku+1]   + 3*q2[ku]  - 6*q2[ku-1]  + q2[ku-2])/(6*dz) + \
                            um*(-2*q2[ku-1]  - 3*q2[ku]  + 6*q2[ku+1]  - q2[ku+2])/(6*dz)
                            
            elif j == 1:
                
                qnl[ku] +=  up*(q2[ku] - q2[ku-1])/dz + \
                            um*(-2*q2[ku-1]  - 3*q2[ku]  + 6*q2[ku+1]  - q2[ku+2])/(6*dz)
                            
            elif j == colsu-2:
                
                qnl[ku] +=  up*(2*q2[ku+1]   + 3*q2[ku]  - 6*q2[ku-1]  + q2[ku-2])/(6*dz) + \
                            um*(q2[ku+1] - q2[ku])/dz
            
            
            # -------------------------------------------------
            # -------- r derivatives --------------------------
            # -------------------------------------------------
            if i > 1 and i < rowsu-2:
                
                f = vps*(2*q2[ku+colsu]   + 3*q2[ku]  - 6*q2[ku-colsu]   + q2[ku-2*colsu])/(6*dr) + \
                    vms*(-2*q2[ku-colsu]  - 3*q2[ku]  + 6*q2[ku+colsu]   - q2[ku+2*colsu])/(6*dr) 
                
                qnl[kv] += 0.25*f*q1[ku]
                qnl[kv+1] += 0.25*f*q1[ku]
                qnl[kv-colsv] += 0.25*f*q1[ku]
                qnl[kv-colsv+1] += 0.25*f*q1[ku]
                    
            elif i == 1:
                
                f = vps*(q2[ku] - q2[ku-colsu])/dr + \
                    vms*(-2*q2[ku-colsu]  - 3*q2[ku]  + 6*q2[ku+colsu]   - q2[ku+2*colsu])/(6*dr) 
                    
                qnl[kv] += 0.25*f*q1[ku]
                qnl[kv+1] += 0.25*f*q1[ku]
                qnl[kv-colsv] += 0.25*f*q1[ku]
                qnl[kv-colsv+1] += 0.25*f*q1[ku]
                            
            elif i == rowsu-2:
                
                f = vps*(2*q2[ku+colsu]   + 3*q2[ku]  - 6*q2[ku-colsu]   + q2[ku-2*colsu])/(6*dr) + \
                    vms*(q2[ku+colsu] - q2[ku])/dr
                    
                qnl[kv] += 0.25*f*q1[ku]
                qnl[kv+1] += 0.25*f*q1[ku]
                qnl[kv-colsv] += 0.25*f*q1[ku]
                qnl[kv-colsv+1] += 0.25*f*q1[ku]
                    

    # -----------------------------------------------
    # ---------- Radial momentum --------------------
    # -----------------------------------------------
                            
    for i in range (1,rowsv-1):
        for j in range (1,colsv-1):
            
            ku = i*colsu + j
            kv = i*colsv + j + szu
            
            u = 0.25*(q1[ku] + q1[ku+colsu] + q1[ku-1] + q1[ku-1+colsu])
            v = q1[kv]
            
            up = u; ups = 1; um = 0; ums = 0
            vp = v; vm = 0
            
            # -------------------------------------------------
            # -------- z derivatives --------------------------
            # -------------------------------------------------
            
            if j > 1 and j < colsv-2:
                
                f = ups*(2*q2[kv+1]   + 3*q2[kv]  - 6*q2[kv-1]  + q2[kv-2])/(6*dz) + \
                    ums*(-2*q2[kv-1]  - 3*q2[kv]  + 6*q2[kv+1]  - q2[kv+2])/(6*dz)
                    
                qnl[ku] += 0.25*f*q1[kv]
                qnl[ku+colsu] += 0.25*f*q1[kv]
                qnl[ku-1] += 0.25*f*q1[kv]
                qnl[ku-1+colsu] += 0.25*f*q1[kv]
                                                        
            elif j == 1:
                
                f = ups*(q2[kv] - q2[kv-1])/dz + \
                    ums*(-2*q2[kv-1]  - 3*q2[kv]  + 6*q2[kv+1]  - q2[kv+2])/(6*dz)
                    
                qnl[ku] += 0.25*f*q1[kv]
                qnl[ku+colsu] += 0.25*f*q1[kv]
                qnl[ku-1] += 0.25*f*q1[kv]
                qnl[ku-1+colsu] += 0.25*f*q1[kv]
                            
            elif j == colsv-2:
                
                f = ups*(2*q2[kv+1]   + 3*q2[kv]  - 6*q2[kv-1]  + q2[kv-2])/(6*dz) + \
                    ums*(q2[kv+1] - q2[kv])/dz
                    
                qnl[ku] += 0.25*f*q1[kv]
                qnl[ku+colsu] += 0.25*f*q1[kv]
                qnl[ku-1] += 0.25*f*q1[kv]
                qnl[ku-1+colsu] += 0.25*f*q1[kv]
                            
            
            # -------------------------------------------------
            # -------- r derivatives --------------------------
            # -------------------------------------------------
            
            if i > 1 and i < rowsv-2:
                
                qnl[kv] +=  vp*(2*q2[kv+colsv]   + 3*q2[kv]  - 6*q2[kv-colsv]   + q2[kv-2*colsv])/(6*dr) + \
                            vm*(-2*q2[kv-colsv]  - 3*q2[kv]  + 6*q2[kv+colsv]   - q2[kv+2*colsv])/(6*dr) 
                            
            elif i == 1:
                
                qnl[kv] +=  vp*(q2[kv] - q2[kv-colsv])/dr + \
                            vm*(-2*q2[kv-colsv]  - 3*q2[kv]  + 6*q2[kv+colsv]   - q2[kv+2*colsv])/(6*dr) 
                            
            elif i == rowsv-2:
                
                qnl[kv] +=  vp*(2*q2[kv+colsv]   + 3*q2[kv]  - 6*q2[kv-colsv]   + q2[kv-2*colsv])/(6*dr) + \
                            vm*(q2[kv+colsv] - q2[kv])/dr
            
    return qnl
            


@numba.njit("f8[:](f8[:],f8[:],f8[:],f8[:])",cache=True) 
def evaluate_linearized_bilinearity(r,z,Q,q):
    
    dr = r[1] - r[0]
    dz = z[1] - z[0]
    
    rowsu = len(r) + 2
    colsu = len(z) + 1
    szu = rowsu*colsu
    
    rowsv = len(r) + 1
    colsv = len(z) + 2
    szv = rowsv*colsv
    
    qnl = np.zeros(szu+szv)
    # -----------------------------------------------
    # ---------- Axial momentum ---------------------
    # -----------------------------------------------
    
    for i in range (1,rowsu-1):
        for j in range (1,colsu-1):
            
            ku = i*colsu + j
            kv = i*colsv + j + szu
            
            u = Q[ku]
            v = 0.25*(Q[kv] + Q[kv+1] + Q[kv-colsv] + Q[kv-colsv+1])
            
            up = u; um = 0
            vp = v; vm = 0
            
            # -------------------------------------------------
            # -------- z derivatives --------------------------
            # -------------------------------------------------
            if j > 1 and j < colsu-2:
            
                qnl[ku] +=  up*(2*q[ku+1]   + 3*q[ku]  - 6*q[ku-1]  + q[ku-2])/(6*dz) + \
                            um*(-2*q[ku-1]  - 3*q[ku]  + 6*q[ku+1]  - q[ku+2])/(6*dz) + \
                            q[ku]*(Q[ku+1] - Q[ku-1])/(2*dz)
                            
            elif j == 1:
                
                qnl[ku] +=  up*(q[ku] - q[ku-1])/dz + \
                            um*(-2*q[ku-1]  - 3*q[ku]  + 6*q[ku+1]  - q[ku+2])/(6*dz) + \
                            q[ku]*(Q[ku+1] - Q[ku-1])/(2*dz)
                            
            elif j == colsu-2:
                
                qnl[ku] +=  up*(2*q[ku+1]   + 3*q[ku]  - 6*q[ku-1]  + q[ku-2])/(6*dz) + \
                            um*(q[ku+1] - q[ku])/dz + q[ku]*(Q[ku+1] - Q[ku-1])/(2*dz)
            
            
            # -------------------------------------------------
            # -------- r derivatives --------------------------
            # -------------------------------------------------
            if i > 1 and i < rowsu-2:
                
                qnl[ku] +=  vp*(2*q[ku+colsu]   + 3*q[ku]  - 6*q[ku-colsu]   + q[ku-2*colsu])/(6*dr) + \
                            vm*(-2*q[ku-colsu]  - 3*q[ku]  + 6*q[ku+colsu]   - q[ku+2*colsu])/(6*dr) + \
                            0.25*(q[kv] + q[kv+1] + q[kv-colsv] + q[kv-colsv+1])*(Q[ku+colsu] - Q[ku-colsu])/(2*dr)
                            
            elif i == 1:
                
                qnl[ku] +=  vp*(q[ku] - q[ku-colsu])/dr + \
                            vm*(-2*q[ku-colsu]  - 3*q[ku]  + 6*q[ku+colsu]   - q[ku+2*colsu])/(6*dr) + \
                            0.25*(q[kv] + q[kv+1] + q[kv-colsv] + q[kv-colsv+1])*(Q[ku+colsu] - Q[ku-colsu])/(2*dr)
                            
            elif i == rowsu-2:
                
                qnl[ku] +=  vp*(2*q[ku+colsu]   + 3*q[ku]  - 6*q[ku-colsu]   + q[ku-2*colsu])/(6*dr) + \
                            vm*(q[ku+colsu] - q[ku])/dr +\
                            0.25*(q[kv] + q[kv+1] + q[kv-colsv] + q[kv-colsv+1])*(Q[ku+colsu] - Q[ku-colsu])/(2*dr)
                            
    
    # -----------------------------------------------
    # ---------- Radial momentum --------------------
    # -----------------------------------------------
                            
    for i in range (1,rowsv-1):
        for j in range (1,colsv-1):
            
            ku = i*colsu + j
            kv = i*colsv + j + szu
            
            u = 0.25*(Q[ku] + Q[ku+colsu] + Q[ku-1] + Q[ku-1+colsu])
            v = Q[kv]
            
            up = u; um = 0
            vp = v; vm = 0
            
            # -------------------------------------------------
            # -------- z derivatives --------------------------
            # -------------------------------------------------
            
            if j > 1 and j < colsv-2:
                
                qnl[kv] +=  up*(2*q[kv+1]   + 3*q[kv]  - 6*q[kv-1]  + q[kv-2])/(6*dz) + \
                            um*(-2*q[kv-1]  - 3*q[kv]  + 6*q[kv+1]  - q[kv+2])/(6*dz) + \
                            0.25*(q[ku] + q[ku+colsu] + q[ku-1] + q[ku-1+colsu])*(Q[kv+1] - Q[kv-1])/(2*dz)
                                                        
            elif j == 1:
                
                qnl[kv] +=  up*(q[kv] - q[kv-1])/dz + \
                            um*(-2*q[kv-1]  - 3*q[kv]  + 6*q[kv+1]  - q[kv+2])/(6*dz) + \
                            0.25*(q[ku] + q[ku+colsu] + q[ku-1] + q[ku-1+colsu])*(Q[kv+1] - Q[kv-1])/(2*dz)
                            
            elif j == colsv-2:
                
                qnl[kv] +=  up*(2*q[kv+1]   + 3*q[kv]  - 6*q[kv-1]  + q[kv-2])/(6*dz) + \
                            um*(q[kv+1] - q[kv])/dz + \
                            0.25*(q[ku] + q[ku+colsu] + q[ku-1] + q[ku-1+colsu])*(Q[kv+1] - Q[kv-1])/(2*dz)
                            
            
            # -------------------------------------------------
            # -------- r derivatives --------------------------
            # -------------------------------------------------
            
            if i > 1 and i < rowsv-2:
                
                qnl[kv] +=  vp*(2*q[kv+colsv]   + 3*q[kv]  - 6*q[kv-colsv]   + q[kv-2*colsv])/(6*dr) + \
                            vm*(-2*q[kv-colsv]  - 3*q[kv]  + 6*q[kv+colsv]   - q[kv+2*colsv])/(6*dr) + \
                            q[kv]*(Q[kv+colsv] - Q[kv-colsv])/(2*dr)
                            
            elif i == 1:
                
                qnl[kv] +=  vp*(q[kv] - q[kv-colsv])/dr + \
                            vm*(-2*q[kv-colsv]  - 3*q[kv]  + 6*q[kv+colsv]   - q[kv+2*colsv])/(6*dr) + \
                            q[kv]*(Q[kv+colsv] - Q[kv-colsv])/(2*dr)
                            
            elif i == rowsv-2:
                
                qnl[kv] +=  vp*(2*q[kv+colsv]   + 3*q[kv]  - 6*q[kv-colsv]   + q[kv-2*colsv])/(6*dr) + \
                            vm*(q[kv+colsv] - q[kv])/dr + q[kv]*(Q[kv+colsv] - Q[kv-colsv])/(2*dr)
            
            
    return qnl

@numba.njit("f8[:](f8[:],f8[:],f8[:],f8[:])",cache=True) 
def evaluate_adjoint_bilinearity(r,z,Q,q):
    
    dr = r[1] - r[0]
    dz = z[1] - z[0]
    
    rowsu = len(r) + 2
    colsu = len(z) + 1
    szu = rowsu*colsu
    
    rowsv = len(r) + 1
    colsv = len(z) + 2
    szv = rowsv*colsv
    
    qnl = np.zeros(szu+szv)
    # -----------------------------------------------
    # ---------- Axial momentum ---------------------
    # -----------------------------------------------
    
    for i in range (1,rowsu-1):
        for j in range (1,colsu-1):
            
            ku = i*colsu + j
            kv = i*colsv + j + szu
            
            u = Q[ku]
            v = 0.25*(Q[kv] + Q[kv+1] + Q[kv-colsv] + Q[kv-colsv+1])
            
            up = u; um = 0
            vp = v; vm = 0
            
            
            # -------------------------------------------------
            # -------- z derivatives --------------------------
            # -------------------------------------------------
            if j > 1 and j < colsu-2:
            
                # Reference indices from the forward simulation
                # qnl[ku] +=  up*(2*q[ku+1]   + 3*q[ku]  - 6*q[ku-1]  + q[ku-2])/(6*dz) + \
                #             um*(-2*q[ku-1]  - 3*q[ku]  + 6*q[ku+1]  - q[ku+2])/(6*dz) + \
                #             q[ku]*(Q[ku+1] - Q[ku-1])/(2*dz)
                            
                qnl[ku] += (up*3*q[ku] - um*3*q[ku])/(6*dz) + q[ku]*(Q[ku+1] - Q[ku-1])/(2*dz)
                qnl[ku+1] += (up*2*q[ku] + um*6*q[ku])/(6*dz)
                qnl[ku-1] += (-um*2*q[ku] - up*6*q[ku])/(6*dz)
                qnl[ku+2] += -um*q[ku]/(6*dz)
                qnl[ku-2] += up*q[ku]/(6*dz)
                
            elif j == 1:
                
                # Reference indices from the forward simulation
                # qnl[ku] +=  up*(q[ku] - q[ku-1])/dz + \
                #             um*(-2*q[ku-1]  - 3*q[ku]  + 6*q[ku+1]  - q[ku+2])/(6*dz) + \
                #             q[ku]*(Q[ku+1] - Q[ku-1])/(2*dz)
                            
                qnl[ku] += up*q[ku]/dz - 3*um*q[ku]/(6*dz) + q[ku]*(Q[ku+1] - Q[ku-1])/(2*dz)
                qnl[ku-1] += -up*q[ku]/dz - 2*um*q[ku]/(6*dz)
                qnl[ku+1] += 6*um*q[ku]/(6*dz)
                qnl[ku+2] += -um*q[ku]/(6*dz)
                            
            elif j == colsu-2:
                
                # Reference indices for the forward simulation
                # qnl[ku] +=  up*(2*q[ku+1]   + 3*q[ku]  - 6*q[ku-1]  + q[ku-2])/(6*dz) + \
                #             um*(q[ku+1] - q[ku])/dz + q[ku]*(Q[ku+1] - Q[ku-1])/(2*dz)
                            
                qnl[ku] += 3*up*q[ku]/(6*dz) - um*q[ku]/dz + q[ku]*(Q[ku+1] - Q[ku-1])/(2*dz)
                qnl[ku-1] += -up*6*q[ku]/(6*dz)
                qnl[ku+1] += um*q[ku]/dz + 2*up*q[ku]/(6*dz)
                qnl[ku-2] += up*q[ku]/(6*dz)
            
            
            # -------------------------------------------------
            # -------- r derivatives --------------------------
            # -------------------------------------------------
            if i > 1 and i < rowsu-2:
                
                # Reference indices for forward simulation
                # qnl[ku] +=  vp*(2*q[ku+colsu]   + 3*q[ku]  - 6*q[ku-colsu]   + q[ku-2*colsu])/(6*dr) + \
                #             vm*(-2*q[ku-colsu]  - 3*q[ku]  + 6*q[ku+colsu]   - q[ku+2*colsu])/(6*dr) + \
                #             0.25*(q[kv] + q[kv+1] + q[kv-colsv] + q[kv-colsv+1])*(Q[ku+colsu] - Q[ku-colsu])/(2*dr)
                            
                qnl[ku] += (3*vp*q[ku] - 3*vm*q[ku])/(6*dr)
                qnl[ku+colsu] += (2*vp*q[ku] + 6*vm*q[ku])/(6*dr)
                qnl[ku-colsu] += (-2*vm*q[ku] - 6*vp*q[ku])/(6*dr)
                qnl[ku+2*colsu] += -vm*q[ku]/(6*dr)
                qnl[ku-2*colsu] += vp*q[ku]/(6*dr)
                qnl[kv] += 0.25*q[ku]*(Q[ku+colsu] - Q[ku-colsu])/(2*dr)
                qnl[kv+1] += 0.25*q[ku]*(Q[ku+colsu] - Q[ku-colsu])/(2*dr)
                qnl[kv-colsv] += 0.25*q[ku]*(Q[ku+colsu] - Q[ku-colsu])/(2*dr)
                qnl[kv-colsv+1] += 0.25*q[ku]*(Q[ku+colsu] - Q[ku-colsu])/(2*dr)
                
                
            elif i == 1:
                
                # Reference indices for forward simulation
                # qnl[ku] +=  vp*(q[ku] - q[ku-colsu])/dr + \
                #             vm*(-2*q[ku-colsu]  - 3*q[ku]  + 6*q[ku+colsu]   - q[ku+2*colsu])/(6*dr) + \
                #             0.25*(q[kv] + q[kv+1] + q[kv-colsv] + q[kv-colsv+1])*(Q[ku+colsu] - Q[ku-colsu])/(2*dr)
                            
                qnl[ku] += vp*q[ku]/dr - 3*vm*q[ku]/(6*dr)
                qnl[ku-colsu] += -vp*q[ku]/dr - 2*vm*q[ku]/(6*dr)
                qnl[ku+colsu] += 6*vm*q[ku]/(6*dr)
                qnl[ku+2*colsu] += -vm*q[ku]/(6*dr)
                qnl[kv] += 0.25*q[ku]*(Q[ku+colsu] - Q[ku-colsu])/(2*dr)
                qnl[kv+1] += 0.25*q[ku]*(Q[ku+colsu] - Q[ku-colsu])/(2*dr)
                qnl[kv-colsv] += 0.25*q[ku]*(Q[ku+colsu] - Q[ku-colsu])/(2*dr)
                qnl[kv-colsv+1] += 0.25*q[ku]*(Q[ku+colsu] - Q[ku-colsu])/(2*dr)
                            
            elif i == rowsu-2:
                
                # Reference indices for forward simulation
                # qnl[ku] +=  vp*(2*q[ku+colsu]   + 3*q[ku]  - 6*q[ku-colsu]   + q[ku-2*colsu])/(6*dr) + \
                #             vm*(q[ku+colsu] - q[ku])/dr +\
                #             0.25*(q[kv] + q[kv+1] + q[kv-colsv] + q[kv-colsv+1])*(Q[ku+colsu] - Q[ku-colsu])/(2*dr)
                            
                qnl[ku] += vp*3*q[ku]/(6*dr) - vm*q[ku]/dr
                qnl[ku+colsu] += 2*vp*q[ku]/(6*dr) + vm*q[ku]/dr
                qnl[ku-colsu] += -6*vp*q[ku]/(6*dr)
                qnl[ku-2*colsu] += vp*q[ku]/(6*dr)
                qnl[kv] += 0.25*q[ku]*(Q[ku+colsu] - Q[ku-colsu])/(2*dr)
                qnl[kv+1] += 0.25*q[ku]*(Q[ku+colsu] - Q[ku-colsu])/(2*dr)
                qnl[kv-colsv] += 0.25*q[ku]*(Q[ku+colsu] - Q[ku-colsu])/(2*dr)
                qnl[kv-colsv+1] += 0.25*q[ku]*(Q[ku+colsu] - Q[ku-colsu])/(2*dr)
                
                            
    
    # -----------------------------------------------
    # ---------- Radial momentum --------------------
    # -----------------------------------------------
                            
    for i in range (1,rowsv-1):
        for j in range (1,colsv-1):
            
            ku = i*colsu + j
            kv = i*colsv + j + szu
            
            u = 0.25*(Q[ku] + Q[ku+colsu] + Q[ku-1] + Q[ku-1+colsu])
            v = Q[kv]
            
            up = u; um = 0
            vp = v; vm = 0
            
            # -------------------------------------------------
            # -------- z derivatives --------------------------
            # -------------------------------------------------
            
            if j > 1 and j < colsv-2:
                
                # qnl[kv] +=  up*(2*q[kv+1]   + 3*q[kv]  - 6*q[kv-1]  + q[kv-2])/(6*dz) + \
                #             um*(-2*q[kv-1]  - 3*q[kv]  + 6*q[kv+1]  - q[kv+2])/(6*dz) + \
                #             0.25*(q[ku] + q[ku+colsu] + q[ku-1] + q[ku-1+colsu])*(Q[kv+1] - Q[kv-1])/(2*dz)
                            
                qnl[kv] += (up*3*q[kv] - um*3*q[kv])/(6*dz)
                qnl[kv+1] += (up*2*q[kv] + um*6*q[kv])/(6*dz)
                qnl[kv-1] += (-um*2*q[kv] - 6*up*q[kv])/(6*dz)
                qnl[kv+2] += -um*q[kv]/(6*dz)
                qnl[kv-2] += up*q[kv]/(6*dz)
                qnl[ku] += 0.25*q[kv]*(Q[kv+1] - Q[kv-1])/(2*dz)
                qnl[ku+colsu] += 0.25*q[kv]*(Q[kv+1] - Q[kv-1])/(2*dz)
                qnl[ku-1] += 0.25*q[kv]*(Q[kv+1] - Q[kv-1])/(2*dz)
                qnl[ku-1+colsu] += 0.25*q[kv]*(Q[kv+1] - Q[kv-1])/(2*dz)
                                                        
            elif j == 1:
                
                # qnl[kv] +=  up*(q[kv] - q[kv-1])/dz + \
                #             um*(-2*q[kv-1]  - 3*q[kv]  + 6*q[kv+1]  - q[kv+2])/(6*dz) + \
                #             0.25*(q[ku] + q[ku+colsu] + q[ku-1] + q[ku-1+colsu])*(Q[kv+1] - Q[kv-1])/(2*dz)
                
                qnl[kv] += up*q[kv]/dz - um*3*q[kv]/(6*dz)
                qnl[kv+1] += um*6*q[kv]/(6*dz)
                qnl[kv-1] += -um*2*q[kv]/(6*dz) - up*q[kv]/dz
                qnl[kv+2] += -um*q[kv]/(6*dz)
                qnl[ku] += 0.25*q[kv]*(Q[kv+1] - Q[kv-1])/(2*dz)
                qnl[ku+colsu] += 0.25*q[kv]*(Q[kv+1] - Q[kv-1])/(2*dz)
                qnl[ku-1] += 0.25*q[kv]*(Q[kv+1] - Q[kv-1])/(2*dz)
                qnl[ku-1+colsu] += 0.25*q[kv]*(Q[kv+1] - Q[kv-1])/(2*dz)
                
                            
            elif j == colsv-2:
                
                # qnl[kv] +=  up*(2*q[kv+1]   + 3*q[kv]  - 6*q[kv-1]  + q[kv-2])/(6*dz) + \
                #             um*(q[kv+1] - q[kv])/dz + \
                #             0.25*(q[ku] + q[ku+colsu] + q[ku-1] + q[ku-1+colsu])*(Q[kv+1] - Q[kv-1])/(2*dz)
                            
                qnl[kv] += up*3*q[kv]/(6*dz) - um*q[kv]/dz
                qnl[kv+1] += up*2*q[kv]/(6*dz) + um*q[kv]/dz
                qnl[kv-1] += - 6*up*q[kv]/(6*dz)
                qnl[kv-2] += up*q[kv]/(6*dz)
                qnl[ku] += 0.25*q[kv]*(Q[kv+1] - Q[kv-1])/(2*dz)
                qnl[ku+colsu] += 0.25*q[kv]*(Q[kv+1] - Q[kv-1])/(2*dz)
                qnl[ku-1] += 0.25*q[kv]*(Q[kv+1] - Q[kv-1])/(2*dz)
                qnl[ku-1+colsu] += 0.25*q[kv]*(Q[kv+1] - Q[kv-1])/(2*dz)
                            
            
            # -------------------------------------------------
            # -------- r derivatives --------------------------
            # -------------------------------------------------
            
            if i > 1 and i < rowsv-2:
                
                # qnl[kv] +=  vp*(2*q[kv+colsv]   + 3*q[kv]  - 6*q[kv-colsv]   + q[kv-2*colsv])/(6*dr) + \
                #             vm*(-2*q[kv-colsv]  - 3*q[kv]  + 6*q[kv+colsv]   - q[kv+2*colsv])/(6*dr) + \
                #             q[kv]*(Q[kv+colsv] - Q[kv-colsv])/(2*dr)
                            
                qnl[kv] += (3*vp*q[kv] - 3*vm*q[kv])/(6*dr) + q[kv]*(Q[kv+colsv] - Q[kv-colsv])/(2*dr)
                qnl[kv+colsv] += (vp*2*q[kv] + 6*vm*q[kv])/(6*dr)
                qnl[kv-colsv] += (-2*vm*q[kv] - 6*vp*q[kv])/(6*dr)
                qnl[kv+2*colsv] += -vm*q[kv]/(6*dr)
                qnl[kv-2*colsv] += vp*q[kv]/(6*dr)
                
                            
            elif i == 1:
                
                # qnl[kv] +=  vp*(q[kv] - q[kv-colsv])/dr + \
                #             vm*(-2*q[kv-colsv]  - 3*q[kv]  + 6*q[kv+colsv] - q[kv+2*colsv])/(6*dr) + \
                #             q[kv]*(Q[kv+colsv] - Q[kv-colsv])/(2*dr)
                            
                qnl[kv] += vp*q[kv]/dr - 3*vm*q[kv]/(6*dr) + q[kv]*(Q[kv+colsv] - Q[kv-colsv])/(2*dr)
                qnl[kv+colsv] += 6*vm*q[kv]/(6*dr)
                qnl[kv-colsv] += -2*vm*q[kv]/(6*dr) - vp*q[kv]/dr
                qnl[kv+2*colsv] += -vm*q[kv]/(6*dr)
                            
            elif i == rowsv-2:
                
                # qnl[kv] +=  vp*(2*q[kv+colsv]   + 3*q[kv]  - 6*q[kv-colsv]   + q[kv-2*colsv])/(6*dr) + \
                #             vm*(q[kv+colsv] - q[kv])/dr + q[kv]*(Q[kv+colsv] - Q[kv-colsv])/(2*dr)
                            
                qnl[kv] += 3*vp*q[kv]/(6*dr) - vm*q[kv]/dr + q[kv]*(Q[kv+colsv] - Q[kv-colsv])/(2*dr)
                qnl[kv+colsv] += vp*2*q[kv]/(6*dr) + vm*q[kv]/dr
                qnl[kv-colsv] += -6*vp*q[kv]/(6*dr)
                qnl[kv-2*colsv] += vp*q[kv]/(6*dr)
            
            
    return qnl
            
            