# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 12:52:25 2020

@author: LENOVO
"""

def insertionSort(nlist):
    for index in range(1,len(nlist)):
        currentvalue = nlist[index] 
        position = index
        while position>0 and nlist[position-1]>currentvalue:
            nlist[position]=nlist[position-1]
            position = position-1
 
            nlist[position]=currentvalue
nlist = [14,46,43,27,57,41,45,21,70]
insertionSort(nlist) 
print(nlist) 
