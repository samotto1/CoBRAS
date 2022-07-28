#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 19:46:56 2021

@author: alberto
"""

import boundary_conditions as bcs
import nonlinear_operators as nonlin_ops
import numpy as np


def nonlinear_solver(jet,lops,time,nsave,qic,*argv):
    
    dt = time[1] - time[0]
    tsave = time[::nsave]
    data = np.zeros((len(qic),len(tsave)))
    
    if len(argv) > 0:   forcing = argv[0]
    else:               forcing = np.zeros(len(time))
    
    q = qic.copy()
    q = bcs.update_boundary_conditions(jet.r,jet.z,q,jet.theta0,flag=1)
    q = q - lops.G.dot(lops.luLp.solve(lops.D.dot(q))).real
    q = bcs.update_boundary_conditions(jet.r,jet.z,q,jet.theta0,flag=1)
    data[:,0] = q
    
    qnl = np.zeros(jet.szu+jet.szv)
    qrhs = np.zeros(jet.szu+jet.szv)
    qstar = np.zeros(jet.szu+jet.szv)
    qrhsfwd = np.zeros(jet.szu+jet.szv)
    qrhsim1 = np.zeros(jet.szu+jet.szv)
    p = np.zeros(jet.szp)
    
    isave = 0
    saveidx = 1
    for i in range (1,len(time)):
        
#        print("Timestep: %d"%i)
        
        q = bcs.update_boundary_conditions(jet.r,jet.z,q,jet.theta0,flag=1)
        qnl[:] = -nonlin_ops.evaluate_bilinearity(jet.r,jet.z,q,q)
        qrhs[:] = qnl + lops.L.dot(q) + lops.B*forcing[i-1]
        
        if i == 1: 
            qrhsfwd[:] = qrhs
            qrhsim1[:]  = qrhs
        else:
            qrhsfwd[:]  = 1.5*qrhs - 0.5*qrhsim1
            qrhsim1[:]  = qrhs

            
        qstar[:] = q + dt*qrhsfwd
        p[:] = lops.luLp.solve(lops.D.dot(qstar)).real
        q[:] = qstar - lops.G.dot(p)
        
        isave += 1
        if isave == nsave:
            
            q = bcs.update_boundary_conditions(jet.r,jet.z,q,jet.theta0,flag=1)
            data[:,saveidx] = q
                
            saveidx += 1
            isave = 0
            
            if (np.linalg.norm(q) > 1e6):
                raise ValueError ("The nonlinear solver blew up.")
            
    return data, tsave
        


def linear_solver(jet,lops,time,nsave,qbflow,qic,flag):
    
    q = qic.copy()
    qstar = np.zeros(jet.szu+jet.szv)
    qrhsfwd = np.zeros(jet.szu+jet.szv)
    qrhsim1 = np.zeros(jet.szu+jet.szv)
    qnl = np.zeros(jet.szu+jet.szv)
    qnl = np.zeros(jet.szu+jet.szv)
    qrhs = np.zeros(jet.szu+jet.szv)
    p = np.zeros(jet.szp)
    
    
    dt = time[1] - time[0]
    tsave = time[::nsave]
    data = np.zeros((jet.szu+jet.szv,len(tsave)))
    data[:,0] = q
    
    if flag == 'ubf':
        
        isave = 0
        saveidx = 1
        for i in range (1,len(time)):
            
            qnl[:] = -nonlin_ops.evaluate_linearized_bilinearity(jet.r,jet.z,qbflow[:,i-1],q)
            qrhs[:] = qnl + lops.L.dot(q)
            
            if i == 1: 
                qrhsfwd[:] = qrhs
                qrhsim1[:] = qrhs
            else:
                qrhsfwd[:] = 1.5*qrhs - 0.5*qrhsim1
                qrhsim1[:] = qrhs
                
            qstar[:] = q + dt*qrhsfwd
            p[:] = lops.luLp.solve(lops.D.dot(qstar))
            q[:] = qstar - lops.G.dot(p)
            q[:] = lops.BC_op.dot(q)
            
            isave += 1
            if isave == nsave:
                
                data[:,saveidx] = q
                saveidx += 1
                isave = 0
                
                if (np.linalg.norm(q) > 1e6):
                    raise ValueError ("The linear solver blew up.")
            
    elif flag == 'sbf':
        
        isave = 0
        saveidx = 1
        for i in range (1,len(time)):
            
            qnl[:] = -nonlin_ops.evaluate_linearized_bilinearity(jet.r,jet.z,qbflow,q)
            qrhs[:] = qnl + lops.L.dot(q)
            
            if i == 1: 
                qrhsfwd[:] = qrhs
                qrhsim1[:] = qrhs
            else:
                qrhsfwd[:] = 1.5*qrhs - 0.5*qrhsim1
                qrhsim1[:] = qrhs
                
            qstar[:] = q + dt*qrhsfwd
            p[:] = lops.luLp.solve(lops.D.dot(qstar))
            q[:] = qstar - lops.G.dot(p)
            q[:] = lops.BC_op.dot(q)
            
            isave += 1
            if isave == nsave:
                
                data[:,saveidx] = q
                saveidx += 1
                isave = 0
                
                if (np.linalg.norm(q) > 1e6):
                    raise ValueError ("The linear solver blew up.")
            
    return data, tsave


def adjoint_solver(jet,lops,time,nsave,qbflow,qic,flag):
    
    q = qic.copy()    
    qstar = np.zeros(jet.szu+jet.szv)
    qrhsfwd = np.zeros(jet.szu+jet.szv)
    qrhsim1 = np.zeros(jet.szu+jet.szv)
    qnl = np.zeros(jet.szu+jet.szv)
    qnl = np.zeros(jet.szu+jet.szv)
    qrhs = np.zeros(jet.szu+jet.szv)
    p = np.zeros(jet.szp)
    
    
    dt = time[1] - time[0]
    
    tsave = time[::nsave]
    data = np.zeros((jet.szu+jet.szv,len(tsave)))
    data[:,0] = q
    
    if flag == 'ubf':
        
        isave = 0
        saveidx = 1
        for i in range (1,len(time)):
            
            q[:] = lops.BC_op_T.dot(lops.W.dot(q))
            if i == 1:  qnl[:] = -nonlin_ops.evaluate_adjoint_bilinearity(jet.r,jet.z,qbflow[:,0],q)
            else:       qnl[:] = -nonlin_ops.evaluate_adjoint_bilinearity(jet.r,jet.z,qbflow[:,-1-(i-2)],q)
                
            qrhs[:] = qnl + lops.LT.dot(q)
            
            if i == 1: 
                qrhsfwd[:] = qrhs
                qrhsim1[:] = qrhs
            else:
                qrhsfwd[:] = 1.5*qrhs - 0.5*qrhsim1
                qrhsim1[:] = qrhs
                
            qstar[:] = q + dt*qrhsfwd
            p[:] = lops.luLpT.solve(lops.GT.dot(qstar))
            q[:] = qstar - lops.DT.dot(p)
            q[:] = lops.BC_op_T.dot(lops.Winv.dot(q))
            
            isave += 1
            if isave == nsave:
                
                data[:,saveidx] = q
                saveidx += 1
                isave = 0
                
                if (np.linalg.norm(q) > 1e6):
                    raise ValueError ("The adjoint solver blew up.")
            
    elif flag == 'sbf':
        
        isave = 0
        saveidx = 1
        for i in range (1,len(time)):
            
            q[:] = lops.BC_op_T.dot(lops.W.dot(q))
            qnl[:] = -nonlin_ops.evaluate_adjoint_bilinearity(jet.r,jet.z,qbflow,q)
                
            qrhs[:] = qnl + lops.LT.dot(q)
            
            if i == 1: 
                qrhsfwd[:] = qrhs
                qrhsim1[:] = qrhs
            else:
                qrhsfwd[:] = 1.5*qrhs - 0.5*qrhsim1
                qrhsim1[:] = qrhs
                
            qstar[:] = q + dt*qrhsfwd
            p[:] = lops.luLpT.solve(lops.GT.dot(qstar))
            q[:] = qstar - lops.DT.dot(p)
            q[:] = lops.BC_op_T.dot(lops.Winv.dot(q))
            
            isave += 1
            if isave == nsave:
                
                data[:,saveidx] = q
                saveidx += 1
                isave = 0
                
                if (np.linalg.norm(q) > 1e6):
                    raise ValueError ("The adjoint solver blew up.")
        
        
    return data, tsave


    
    
    
    
