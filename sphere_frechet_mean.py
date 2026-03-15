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
        if distance == 0:
            return np.zeros(x.shape)
        direction = self.proj(x,y)
        pytha = 0
        for i in range(len(direction)):
            pytha += ((direction[i])**2)
        normalised = direction / (pytha) ** 0.5
        return normalised * distance
    
    def exp(self, x, v):
        distance = 0
        for i in range(len(v)):
            distance += v[i] ** 2
        magnitude = distance ** 0.5
        if magnitude == 0:
            return x
        normalised_v = v / magnitude 
        return (np.cos(magnitude) * x) + (np.sin(magnitude) * normalised_v)
    
    def frechet_mean(self, data_points, max_iter = 100):
        mu = data_points[0]
        
        for iterations in range(max_iter):
            tangent_vectors = []
            for i in range(len(data_points)):
                tangent_vectors.append(self.log(mu, data_points[i]))
            
            e1 = e2 = e3 = 0 #averaged unit vectors    
            for j in range(len(tangent_vectors)):
                e1 += tangent_vectors[j][0]
                e2 += tangent_vectors[j][1]
                e3 += tangent_vectors[j][2]
            
            averaged_vector = np.array([e1 / (len(data_points)), e2 /  (len(data_points)), e3 / (len(data_points))])
            
            sum_sq= 0
            for l in range(len(averaged_vector)):
                sum_sq += averaged_vector[l] ** 2
            magnitude = sum_sq ** 0.5
            
            if magnitude < 1e-9:
                break
            mu = self.exp(mu, averaged_vector)
        return mu

        
        
        
        
        
        
        
        
        
        
        
        