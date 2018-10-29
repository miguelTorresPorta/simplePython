#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 10:41:05 2018

@author: migueltorresporta
"""

list_numbers = [10,5,3,7]
k = 17

def sum_is_list(list_number, numberK):
    list_number.sort()
    
    i = 0
    j = len(list_number) - 1
    
    while i < j:
        suma = list_number[i] + list_number[j]
        if suma < numberK:
            i = i + 1
        elif suma > numberK:
            j = j - 1
        elif suma == numberK:
            print ("Sum found in the array")
            return True
        
            
        
sum_is_list(list_numbers, k)