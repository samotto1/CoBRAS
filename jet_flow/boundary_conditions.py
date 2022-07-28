#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 19:33:17 2021

@author: alberto
"""

import numpy as np
import numba



@numba.njit("f8[:](f8[:],f8[:],f8[:],f8,i8)",cache=True) 
def update_boundary_conditions(r,z,q,theta0,flag):
        
    rowsu = len(r) + 2
    colsu = len(z) + 1
    szu = rowsu*colsu
    
    rowsv = len(r) + 1
    colsv = len(z) + 2
        
    # -----------------------------------------------------
    # --------- Inflow/Outflow boundaries -----------------
    # -----------------------------------------------------
    
    for i in range (1,rowsu-1):
        ku_in = i*colsu
        ku_out = (i+1)*colsu - 1
        
        if flag == 1:   val = 0.5*(1 - np.tanh((r[i-1] - 1/(4*r[i-1]))/(4*theta0)))
        if flag == 0:   val = 0.0
        
        q[ku_in] = val 
        q[ku_out] = q[ku_out-1] 
        
    for i in range (1,rowsv-1):
        
        kv_in = i*colsv + szu
        kv_out = (i+1)*colsv - 1 + szu
        ku_out = (i+1)*colsu - 1
        
        
        q[kv_in] = -q[kv_in+1]
        q[kv_out] = q[kv_out-1]
        
    # -----------------------------------------------------
    # --------- Top/Bottom boundaries ---------------------
    # -----------------------------------------------------
    for j in range (0,colsu):
        ku_bot = j
        ku_top = (rowsu-1)*colsu + j
        
        q[ku_bot] = q[ku_bot+colsu]
        q[ku_top] = q[ku_top-colsu]
        
    for j in range (0,colsv):
        kv_bot = j + szu
        kv_top = (rowsv-1)*colsv + j + szu
        
        q[kv_bot] = 0.0
        q[kv_top] = q[kv_top-colsv]
        

    return q
    
