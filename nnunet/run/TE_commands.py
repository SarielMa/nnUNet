# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 19:05:33 2023

@author: linhai
"""

import os

os.system("python run_TE_training_v2.py --epsilon 5.0 --task 002 --cuda_id 0")
os.system("python run_TE_training_v2.py --epsilon 10.0 --task 002 --cuda_id 0")
os.system("python run_TE_training_v2.py --epsilon 15.0 --task 002 --cuda_id 0")

