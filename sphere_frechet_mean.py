#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 14:11:21 2026

@author: alexandre
"""
import numpy as np
class Manifold:
    def dist(self, x, y):
        raise NotImplementedError("Every manifold must define a distance metric.")

    def exp(self, x, v):
        raise NotImplementedError("Every manifold must define an exponential map.")
    
    def proj(self, x, v):
        raise NotImplementedError("Every manifold must define a projection map.")
    
    def log(self, x, y):
        raise NotImplementedError("Every manifold must define a logarithmic map.")
    
    

class Sphere(Manifold):
    def dist(self, x, y):
        dot_product = np.clip(np.dot(x,y), -1,1)
        return np.arccos(dot_product)
    
    def proj(self, x, v):
        radial_length = np.dot(x, v)
        radial_vector = radial_length * x
        tangent = v - radial_vector
        return tangent
    
    def log(self, x, y):
        distance = self.dist(x, y)
        
        if distance < 1e-15:
            return np.zeros_like(x)
        
        direction = self.proj(x,y)
        magnitude = np.linalg.norm(direction)
        normalised = direction / magnitude
        
        return normalised * distance
    
    def exp(self, x, v):
 
        magnitude = np.linalg.norm(v)
        if magnitude < 1e-15:
            return x
        normalised_v = v / magnitude 
        return (np.cos(magnitude) * x) + (np.sin(magnitude) * normalised_v)
    

    def frechet_mean(self, data_points, max_iter = 100):
        
        mu = data_points[0]
        
        for iterations in range(max_iter):
            
            tangent_vectors = [ self.log(mu, p) for p in data_points]
            
            averaged_vector = np.mean(tangent_vectors, axis =0)
            
            if np.linalg.norm(averaged_vector) < 1e-9:
                break
            
            mu = self.exp(mu, averaged_vector)
        
        return mu
        
        

        
        
        