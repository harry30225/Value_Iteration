import cvxpy as cp
import numpy as np
import os
import json

np.set_printoptions(threshold=np.inf)

# Data of states
POS_NUM = 5       # W,N,S,E,C
MAT_NUM = 3       # 0,1,2
ARR_NUM = 4       # 0,1,2,3
STATE_NUM = 2     # D,R
HEALTH_NUM = 5    # 0,25,50,75,100
TOTAL_NUM_STATE = POS_NUM * MAT_NUM * ARR_NUM * STATE_NUM * HEALTH_NUM

POS_DIR = ["W","N","E","S","C"]
STATEMM_ARR = ["D","R"]
HEALTH_ARR = [0,25,50,75,100]

ACTIONS = {"NOOP" : 0 , "UP" : 1 , "LEFT" : 2 , "DOWN" : 3, "RIGHT" : 4, "STAY" : 5 , "SHOOT" : 6 , "HIT" : 7 , "CRAFT" : 8 , "GATHER" : 9}


step_cost = -10
stay_cost = -10
mm_hit_reward = -40

dmg_arr = -25
dmg_bld = -50

# at center
pC_succmov = 0.85 # MM in D and R state respectively
pC_hit_arr = 0.5
pC_hit_bld = 0.1
# at north
pN_succmov = 0.85
pN_1arrmk = 0.5
pN_2arrmk = 0.35
pN_3arrmk = 0.15

# at south 
pS_succmov = 0.85
p_succ_getmat = 0.75

# at east
pE_succmov = 1 # MM in D and R state respectively
pE_hit_arr = 0.9
pE_hit_bld = 0.2

# at west
pW_succmov = 1
pW_hit_arr = 0.25

# mm
p_dtor = 0.2
p_stayd = 0.8
p_attack = 0.5
p_noAtk = 0.5

# current
current_pos = "C"
current_mat = 2
current_arrow = 3
current_state = "R"
current_health = 100

class State:
    def __init__(self, pos,mat,arr,state,health):
        self.pos = pos
        self.mat = mat
        self.arr = arr
        self.state = state
        self.health = health

    def CondAction(self , action):
        if action == 0:
            return self.health == 0

        if action == 1 and self.health > 0:
            return (self.pos == "C" or self.pos == "S")

        if action == 2 and self.health > 0:
            return (self.pos == "C" or self.pos == "E")
        
        if action == 3 and self.health > 0:
            return (self.pos == "C" or self.pos == "N")

        if action == 4 and self.health > 0:
            return (self.pos == "C" or self.pos == "W")

        if action == 5 and self.health > 0:
            return 1

        if action == 6 and self.health > 0:
            return (self.arr > 0 and(self.pos == "C" or self.pos == "E" or self.pos == "W"))

        if action == 7 and self.health > 0:
            return (self.pos == "C" or self.pos == "E")

        if action == 8 and self.health > 0:
            return (self.pos == "N" and self.mat > 0)

        if action == 9 and self.health > 0:
            return (self.pos == "S")

    def Actions(self):
        Actions = []
        for i in range(len(ACTIONS)):
            if self.CondAction(i) == 1:
                Actions.append(i)

        return Actions

    def simulate(self,action):
        # action matches condition
        if action in self.Actions():
            if action == 0:
                return []
            
            if action == 1:
                if self.pos == "C":
                    if self.state == "D":
                        s_dd1 = State("N" , self.mat , self.arr,self.state,self.health)
                        s_dd2 = State("E" , self.mat , self.arr,self.state,self.health)
                        s_dr1 = State("N" , self.mat , self.arr,"R",self.health)
                        s_dr2 = State("E" , self.mat , self.arr,"R",self.health)
                        return [
                            (pC_succmov*p_stayd , s_dd1),
                            ((1 - pC_succmov)*p_stayd , s_dd2),
                            (pC_succmov*p_dtor , s_dr1),
                            ((1 - pC_succmov)*p_dtor , s_dr2)
                        ]

                    if self.state == "R":
                        s_ra1 = State("C" , self.mat , 0,"D",min(self.health + 25 , 100))
                        s_rn1 = State("N" , self.mat , self.arr,self.state,self.health)
                        s_rn2 = State("E" , self.mat , self.arr,self.state,self.health)
                        return [
                            (p_attack,s_ra1),
                            (pC_succmov*p_noAtk,s_rn1),
                            ((1 - pC_succmov)*p_noAtk,s_rn2)
                        ]
                    

                if self.pos == "S":
                    if self.state == "D":
                        s_dd1 = State("C" , self.mat , self.arr,self.state,self.health)
                        s_dd2 = State("E" , self.mat , self.arr,self.state,self.health)
                        s_dr1 = State("C" , self.mat , self.arr,"R",self.health)
                        s_dr2 = State("E" , self.mat , self.arr,"R",self.health)
                        return [
                            (pS_succmov*p_stayd , s_dd1),
                            ((1 - pS_succmov)*p_stayd , s_dd2),
                            (pS_succmov*p_dtor , s_dr1),
                            ((1 - pS_succmov)*p_dtor , s_dr2)
                        ]
                    
                    if self.state == "R":
                        s_ra1 = State("C" , self.mat , self.arr,"D",self.health)
                        s_ra2 = State("E" , self.mat , self.arr,"D",self.health)
                        s_rn1 = State("C" , self.mat , self.arr,self.state,self.health)
                        s_rn2 = State("E" , self.mat , self.arr,self.state,self.health)
                        return [
                            (p_attack * (pS_succmov),s_ra1),
                            (p_attack * (1 - pS_succmov),s_ra2),
                            (pS_succmov*p_noAtk,s_rn1),
                            ((1 - pS_succmov)*p_noAtk,s_rn2)
                        ]

            if action == 2:
                if self.pos == "C":
                    if self.state == "D":
                        s_dd1 = State("W" , self.mat , self.arr,self.state,self.health)
                        s_dd2 = State("E" , self.mat , self.arr,self.state,self.health)
                        s_dr1 = State("W" , self.mat , self.arr,"R",self.health)
                        s_dr2 = State("E" , self.mat , self.arr,"R",self.health)
                        return [
                            (pC_succmov*p_stayd , s_dd1),
                            ((1 - pC_succmov)*p_stayd , s_dd2),
                            (pC_succmov*p_dtor , s_dr1),
                            ((1 - pC_succmov)*p_dtor , s_dr2)
                        ]

                    if self.state == "R":
                        s_ra1 = State("C" , self.mat , 0,"D",min(self.health + 25 , 100))
                        s_rn1 = State("W" , self.mat , self.arr,self.state,self.health)
                        s_rn2 = State("E" , self.mat , self.arr,self.state,self.health)
                        return [
                            (p_attack,s_ra1),
                            (pC_succmov*p_noAtk,s_rn1),
                            ((1 - pC_succmov)*p_noAtk,s_rn2)
                        ]

                if self.pos == "E":
                    if self.state == "D":
                        s_dd1 = State("C" , self.mat , self.arr,self.state,self.health)
                        s_dd2 = State("E" , self.mat , self.arr,self.state,self.health)
                        s_dr1 = State("C" , self.mat , self.arr,"R",self.health)
                        s_dr2 = State("E" , self.mat , self.arr,"R",self.health)
                        return [
                            (pE_succmov*p_stayd , s_dd1),
                            ((1 - pE_succmov)*p_stayd , s_dd2),
                            (pE_succmov*p_dtor , s_dr1),
                            ((1 - pE_succmov)*p_dtor , s_dr2)
                        ]

                    if self.state == "R":
                        s_ra1 = State("E" , self.mat , 0,"D",min(self.health + 25 , 100))
                        s_rn1 = State("C" , self.mat , self.arr,self.state,self.health)
                        s_rn2 = State("E" , self.mat , self.arr,self.state,self.health)
                        return [
                            (p_attack,s_ra1),
                            (pE_succmov*p_noAtk,s_rn1),
                            ((1 - pE_succmov)*p_noAtk,s_rn2)
                        ]

            if action == 3:
                if self.pos == "C":
                    if self.state == "D":
                        s_dd1 = State("S" , self.mat , self.arr,self.state,self.health)
                        s_dd2 = State("E" , self.mat , self.arr,self.state,self.health)
                        s_dr1 = State("S" , self.mat , self.arr,"R",self.health)
                        s_dr2 = State("E" , self.mat , self.arr,"R",self.health)
                        return [
                            (pC_succmov*p_stayd , s_dd1),
                            ((1 - pC_succmov)*p_stayd , s_dd2),
                            (pC_succmov*p_dtor , s_dr1),
                            ((1 - pC_succmov)*p_dtor , s_dr2)
                        ]

                    if self.state == "R":
                        s_ra1 = State("C" , self.mat , 0,"D",min(self.health + 25 , 100))
                        s_rn1 = State("S" , self.mat , self.arr,self.state,self.health)
                        s_rn2 = State("E" , self.mat , self.arr,self.state,self.health)
                        return [
                            (p_attack,s_ra1),
                            (pC_succmov*p_noAtk,s_rn1),
                            ((1 - pC_succmov)*p_noAtk,s_rn2)
                        ]
                    

                if self.pos == "N":
                    if self.state == "D":
                        s_dd1 = State("C" , self.mat , self.arr,self.state,self.health)
                        s_dd2 = State("E" , self.mat , self.arr,self.state,self.health)
                        s_dr1 = State("C" , self.mat , self.arr,"R",self.health)
                        s_dr2 = State("E" , self.mat , self.arr,"R",self.health)
                        return [
                            (pN_succmov*p_stayd , s_dd1),
                            ((1 - pN_succmov)*p_stayd , s_dd2),
                            (pN_succmov*p_dtor , s_dr1),
                            ((1 - pN_succmov)*p_dtor , s_dr2)
                        ]
                    
                    if self.state == "R":
                        s_ra1 = State("C" , self.mat , self.arr,"D",self.health)
                        s_ra2 = State("E" , self.mat , self.arr,"D",self.health)
                        s_rn1 = State("C" , self.mat , self.arr,self.state,self.health)
                        s_rn2 = State("E" , self.mat , self.arr,self.state,self.health)
                        return [
                            (p_attack * (pN_succmov),s_ra1),
                            (p_attack * (1 - pN_succmov),s_ra2),
                            (pN_succmov*p_noAtk,s_rn1),
                            ((1 - pN_succmov)*p_noAtk,s_rn2)
                        ]

            if action == 4:
                if self.pos == "C":
                    if self.state == "D":
                        s_dd1 = State("E" , self.mat , self.arr,self.state,self.health)
                        s_dd2 = State("E" , self.mat , self.arr,self.state,self.health)
                        s_dr1 = State("E" , self.mat , self.arr,"R",self.health)
                        s_dr2 = State("E" , self.mat , self.arr,"R",self.health)
                        return [
                            (pC_succmov*p_stayd , s_dd1),
                            ((1 - pC_succmov)*p_stayd , s_dd2),
                            (pC_succmov*p_dtor , s_dr1),
                            ((1 - pC_succmov)*p_dtor , s_dr2)
                        ]

                    if self.state == "R":
                        s_ra1 = State("C" , self.mat , 0,"D",min(self.health + 25 , 100))
                        s_rn1 = State("E" , self.mat , self.arr,self.state,self.health)
                        s_rn2 = State("E" , self.mat , self.arr,self.state,self.health)
                        return [
                            (p_attack,s_ra1),
                            (pC_succmov*p_noAtk,s_rn1),
                            ((1 - pC_succmov)*p_noAtk,s_rn2)
                        ]

                if self.pos == "W":
                    if self.state == "D":
                        s_dd1 = State("C" , self.mat , self.arr,self.state,self.health)
                        s_dd2 = State("E" , self.mat , self.arr,self.state,self.health)
                        s_dr1 = State("C" , self.mat , self.arr,"R",self.health)
                        s_dr2 = State("E" , self.mat , self.arr,"R",self.health)
                        return [
                            (pW_succmov*p_stayd , s_dd1),
                            ((1 - pW_succmov)*p_stayd , s_dd2),
                            (pW_succmov*p_dtor , s_dr1),
                            ((1 - pW_succmov)*p_dtor , s_dr2)
                        ]

                    if self.state == "R":
                        s_ra1 = State("C" , self.mat , self.arr,"D",self.health)
                        s_ra2 = State("E" , self.mat , self.arr,"D",self.health)
                        s_rn1 = State("C" , self.mat , self.arr,self.state,self.health)
                        s_rn2 = State("E" , self.mat , self.arr,self.state,self.health)
                        return [
                            (p_attack*(pW_succmov),s_ra1),
                            (p_attack * (1 - pW_succmov),s_ra2),
                            (pW_succmov*p_noAtk,s_rn1),
                            ((1 - pW_succmov)*p_noAtk,s_rn2)
                        ]

            if action == 5:
                if self.pos == "C":
                    if self.state == "D":
                        s_dd1 = State("C" , self.mat , self.arr,self.state,self.health)
                        s_dd2 = State("E" , self.mat , self.arr,self.state,self.health)
                        s_dr1 = State("C" , self.mat , self.arr,"R",self.health)
                        s_dr2 = State("E" , self.mat , self.arr,"R",self.health)
                        return [
                            (pC_succmov*p_stayd , s_dd1),
                            ((1 - pC_succmov)*p_stayd , s_dd2),
                            (pC_succmov*p_dtor , s_dr1),
                            ((1 - pC_succmov)*p_dtor , s_dr2)
                        ]

                    if self.state == "R":
                        s_ra1 = State("C" , self.mat , 0,"D",min(self.health + 25 , 100))
                        s_rn1 = State("C" , self.mat , self.arr,self.state,self.health)
                        s_rn2 = State("E" , self.mat , self.arr,self.state,self.health)
                        return [
                            (p_attack,s_ra1),
                            (pC_succmov*p_noAtk,s_rn1),
                            ((1 - pC_succmov)*p_noAtk,s_rn2)
                        ]
                
                if self.pos == "N":
                    if self.state == "D":
                        s_dd1 = State("N" , self.mat , self.arr,self.state,self.health)
                        s_dd2 = State("E" , self.mat , self.arr,self.state,self.health)
                        s_dr1 = State("N" , self.mat , self.arr,"R",self.health)
                        s_dr2 = State("E" , self.mat , self.arr,"R",self.health)
                        return [
                            (pN_succmov*p_stayd , s_dd1),
                            ((1 - pN_succmov)*p_stayd , s_dd2),
                            (pN_succmov*p_dtor , s_dr1),
                            ((1 - pN_succmov)*p_dtor , s_dr2)
                        ]
                    
                    if self.state == "R":
                        s_ra1 = State("N" , self.mat , self.arr,"D",self.health)
                        s_ra2 = State("E" , self.mat , self.arr,"D",self.health)
                        s_rn1 = State("N" , self.mat , self.arr,self.state,self.health)
                        s_rn2 = State("E" , self.mat , self.arr,self.state,self.health)
                        return [
                            (p_attack * (pN_succmov),s_ra1),
                            (p_attack * (1 - pN_succmov),s_ra2),
                            (pN_succmov*p_noAtk,s_rn1),
                            ((1 - pN_succmov)*p_noAtk,s_rn2)
                        ]

                if self.pos == "S":
                    if self.state == "D":
                        s_dd1 = State("S" , self.mat , self.arr,self.state,self.health)
                        s_dd2 = State("E" , self.mat , self.arr,self.state,self.health)
                        s_dr1 = State("S" , self.mat , self.arr,"R",self.health)
                        s_dr2 = State("E" , self.mat , self.arr,"R",self.health)
                        return [
                            (pS_succmov*p_stayd , s_dd1),
                            ((1 - pS_succmov)*p_stayd , s_dd2),
                            (pS_succmov*p_dtor , s_dr1),
                            ((1 - pS_succmov)*p_dtor , s_dr2)
                        ]
                    
                    if self.state == "R":
                        s_ra1 = State("S" , self.mat , self.arr,"D",self.health)
                        s_ra2 = State("E" , self.mat , self.arr,"D",self.health)
                        s_rn1 = State("S" , self.mat , self.arr,self.state,self.health)
                        s_rn2 = State("E" , self.mat , self.arr,self.state,self.health)
                        return [
                            (p_attack * (pS_succmov),s_ra1),
                            (p_attack * (1 - pS_succmov),s_ra2),
                            (pS_succmov*p_noAtk,s_rn1),
                            ((1 - pS_succmov)*p_noAtk,s_rn2)
                        ]

                if self.pos == "W":
                    if self.state == "D":
                        s_dd1 = State("W" , self.mat , self.arr,self.state,self.health)
                        s_dd2 = State("E" , self.mat , self.arr,self.state,self.health)
                        s_dr1 = State("W" , self.mat , self.arr,"R",self.health)
                        s_dr2 = State("E" , self.mat , self.arr,"R",self.health)
                        return [
                            (pW_succmov*p_stayd , s_dd1),
                            ((1 - pW_succmov)*p_stayd , s_dd2),
                            (pW_succmov*p_dtor , s_dr1),
                            ((1 - pW_succmov)*p_dtor , s_dr2)
                        ]

                    if self.state == "R":
                        s_ra1 = State("W" , self.mat , self.arr,"D",self.health)
                        s_ra2 = State("E" , self.mat , self.arr,"D",self.health)
                        s_rn1 = State("W" , self.mat , self.arr,self.state,self.health)
                        s_rn2 = State("E" , self.mat , self.arr,self.state,self.health)
                        return [
                            (p_attack*(pW_succmov),s_ra1),
                            (p_attack * (1 - pW_succmov),s_ra2),
                            (pW_succmov*p_noAtk,s_rn1),
                            ((1 - pW_succmov)*p_noAtk,s_rn2)
                        ]

                if self.pos == "E":
                    if self.state == "D":
                        s_dd1 = State("E" , self.mat , self.arr,self.state,self.health)
                        s_dd2 = State("E" , self.mat , self.arr,self.state,self.health)
                        s_dr1 = State("E" , self.mat , self.arr,"R",self.health)
                        s_dr2 = State("E" , self.mat , self.arr,"R",self.health)
                        return [
                            (pE_succmov*p_stayd , s_dd1),
                            ((1 - pE_succmov)*p_stayd , s_dd2),
                            (pE_succmov*p_dtor , s_dr1),
                            ((1 - pE_succmov)*p_dtor , s_dr2)
                        ]

                    if self.state == "R":
                        s_ra1 = State("E" , self.mat , 0,"D",min(self.health + 25 , 100))
                        s_rn1 = State("E" , self.mat , self.arr,self.state,self.health)
                        s_rn2 = State("E" , self.mat , self.arr,self.state,self.health)
                        return [
                            (p_attack,s_ra1),
                            (pE_succmov*p_noAtk,s_rn1),
                            ((1 - pE_succmov)*p_noAtk,s_rn2)
                        ]

            if action == 6:
                if self.pos == "C":
                    if self.state == "D":
                        s_dd1 = State("C" , self.mat , self.arr - 1,self.state,max(self.health - 25 , 0))
                        s_dd2 = State("C" , self.mat , self.arr - 1,self.state,self.health)
                        s_dr1 = State("C" , self.mat , self.arr - 1,"R",max(self.health - 25 , 0))
                        s_dr2 = State("C" , self.mat , self.arr - 1,"R",self.health)
                        return [
                            (pC_hit_arr*p_stayd , s_dd1),
                            ((1 - pC_hit_arr)*p_stayd , s_dd2),
                            (pC_hit_arr*p_dtor , s_dr1),
                            ((1 - pC_hit_arr)*p_dtor , s_dr2)
                        ]

                    if self.state == "R":
                        s_ra1 = State("C" , self.mat , 0,"D",min(self.health + 25 , 100))
                        s_rn1 = State("C" , self.mat , self.arr - 1,self.state,max(self.health - 25 , 0))
                        s_rn2 = State("C" , self.mat , self.arr - 1,self.state,self.health)
                        return [
                            (p_attack,s_ra1),
                            (pC_hit_arr*p_noAtk,s_rn1),
                            ((1 - pC_hit_arr)*p_noAtk,s_rn2)
                        ]

                if self.pos == "E":
                    if self.state == "D":
                        s_dd1 = State("E" , self.mat , self.arr - 1,self.state,max(self.health - 25,0))
                        s_dd2 = State("E" , self.mat , self.arr - 1,self.state,self.health)
                        s_dr1 = State("E" , self.mat , self.arr - 1,"R",max(self.health - 25,0))
                        s_dr2 = State("E" , self.mat , self.arr - 1,"R",self.health)
                        return [
                            (pE_hit_arr*p_stayd , s_dd1),
                            ((1 - pE_hit_arr)*p_stayd , s_dd2),
                            (pE_hit_arr*p_dtor , s_dr1),
                            ((1 - pE_hit_arr)*p_dtor , s_dr2)
                        ]

                    if self.state == "R":
                        s_ra1 = State("E" , self.mat , 0,"D",min(self.health + 25 , 100))
                        s_rn1 = State("E" , self.mat , self.arr - 1,self.state,max(self.health - 25 , 0))
                        s_rn2 = State("E" , self.mat , self.arr - 1,self.state,self.health)
                        return [
                            (p_attack,s_ra1),
                            (pE_hit_arr*p_noAtk,s_rn1),
                            ((1 - pE_hit_arr)*p_noAtk,s_rn2)
                        ]

                if self.pos == "W":
                    if self.state == "D":
                        s_dd1 = State("W" , self.mat , self.arr - 1,self.state,max(self.health - 25,0))
                        s_dd2 = State("W" , self.mat , self.arr - 1,self.state,self.health)
                        s_dr1 = State("W" , self.mat , self.arr - 1,"R",max(self.health - 25,0))
                        s_dr2 = State("W" , self.mat , self.arr - 1,"R",self.health)
                        return [
                            (pW_hit_arr*p_stayd , s_dd1),
                            ((1 - pW_hit_arr)*p_stayd , s_dd2),
                            (pW_hit_arr*p_dtor , s_dr1),
                            ((1 - pW_hit_arr)*p_dtor , s_dr2)
                        ]

                    if self.state == "R":
                        s_ra1 = State("W" , self.mat , self.arr - 1,"D",max(self.health - 25,0))
                        s_ra2 = State("W" , self.mat , self.arr - 1,"D",self.health)
                        s_rn1 = State("W" , self.mat , self.arr - 1,self.state,max(self.health - 25,0))
                        s_rn2 = State("W" , self.mat , self.arr - 1,self.state,self.health)
                        return [
                            (p_attack*(pW_hit_arr),s_ra1),
                            (p_attack * (1 - pW_hit_arr),s_ra2),
                            (pW_hit_arr*p_noAtk,s_rn1),
                            ((1 - pW_hit_arr)*p_noAtk,s_rn2)
                        ]

            if action == 7:
                if self.pos == "C":
                    if self.state == "D":
                        s_dd1 = State("C" , self.mat , self.arr ,self.state,max(self.health - 50 , 0))
                        s_dd2 = State("C" , self.mat , self.arr,self.state,self.health)
                        s_dr1 = State("C" , self.mat , self.arr ,"R",max(self.health - 50 , 0))
                        s_dr2 = State("C" , self.mat , self.arr,"R",self.health)
                        return [
                            (pC_hit_bld*p_stayd , s_dd1),
                            ((1 - pC_hit_bld)*p_stayd , s_dd2),
                            (pC_hit_bld*p_dtor , s_dr1),
                            ((1 - pC_hit_bld)*p_dtor , s_dr2)
                        ]

                    if self.state == "R":
                        s_ra1 = State("C" , self.mat , 0,"D",min(self.health + 25 , 100))
                        s_rn1 = State("C" , self.mat , self.arr ,self.state,max(self.health - 50 , 0))
                        s_rn2 = State("C" , self.mat , self.arr,self.state,self.health)
                        return [
                            (p_attack,s_ra1),
                            (pC_hit_bld*p_noAtk,s_rn1),
                            ((1 - pC_hit_bld)*p_noAtk,s_rn2)
                        ]

                if self.pos == "E":
                    if self.state == "D":
                        s_dd1 = State("E" , self.mat , self.arr ,self.state,max(self.health - 50,0))
                        s_dd2 = State("E" , self.mat , self.arr,self.state,self.health)
                        s_dr1 = State("E" , self.mat , self.arr ,"R",max(self.health - 50,0))
                        s_dr2 = State("E" , self.mat , self.arr,"R",self.health)
                        return [
                            (pE_hit_bld*p_stayd , s_dd1),
                            ((1 - pE_hit_bld)*p_stayd , s_dd2),
                            (pE_hit_bld*p_dtor , s_dr1),
                            ((1 - pE_hit_bld)*p_dtor , s_dr2)
                        ]

                    if self.state == "R":
                        s_ra1 = State("E" , self.mat , 0,"D",min(self.health + 25 , 100))
                        s_rn1 = State("E" , self.mat , self.arr ,self.state,max(self.health - 50 , 0))
                        s_rn2 = State("E" , self.mat , self.arr,self.state,self.health)
                        return [
                            (p_attack,s_ra1),
                            (pE_hit_bld*p_noAtk,s_rn1),
                            ((1 - pE_hit_bld)*p_noAtk,s_rn2)
                        ]

            if action == 8:
                if self.state == "D":
                    s_dd1 = State(self.pos , self.mat - 1 , min(3 , self.arr + 1),self.state,self.health)
                    s_dd2 = State(self.pos , self.mat - 1 , min(3 , self.arr + 2),self.state,self.health)
                    s_dd3 = State(self.pos , self.mat - 1 , min(3 , self.arr + 3),self.state,self.health)
                    s_dr1 = State(self.pos , self.mat - 1 , min(3 , self.arr + 1),"R",self.health)
                    s_dr2 = State(self.pos , self.mat - 1 , min(3 , self.arr + 2),"R",self.health)
                    s_dr3 = State(self.pos , self.mat - 1 , min(3 , self.arr + 3),"R",self.health)
                    return [
                        (p_stayd*pN_1arrmk,s_dd1),
                        (p_stayd*pN_2arrmk,s_dd2),
                        (p_stayd*pN_3arrmk,s_dd3),
                        (p_dtor*pN_1arrmk,s_dr1),
                        (p_dtor*pN_2arrmk,s_dr2),
                        (p_dtor*pN_3arrmk,s_dr3)
                    ]
                
                if self.state == "R":
                    s_ra1 = State(self.pos , self.mat - 1 , min(3 , self.arr + 1),self.state,self.health)
                    s_ra2 = State(self.pos , self.mat - 1 , min(3 , self.arr + 2),self.state,self.health)
                    s_ra3 = State(self.pos , self.mat - 1 , min(3 , self.arr + 3),self.state,self.health)
                    s_rn1 = State(self.pos , self.mat - 1 , min(3 , self.arr + 1),"D",self.health)
                    s_rn2 = State(self.pos , self.mat - 1 , min(3 , self.arr + 2),"D",self.health)
                    s_rn3 = State(self.pos , self.mat - 1 , min(3 , self.arr + 3),"D",self.health)
                    return [
                        (p_attack*pN_1arrmk,s_ra1),
                        (p_attack*pN_2arrmk,s_ra2),
                        (p_attack*pN_3arrmk,s_ra3),
                        (p_noAtk*pN_1arrmk,s_rn1),
                        (p_noAtk*pN_2arrmk,s_rn2),
                        (p_noAtk*pN_3arrmk,s_rn3)
                    ]

            if action == 9:
                if self.state == "D":
                    s_dd1 = State(self.pos , min(2,self.mat + 1),self.arr,self.state,self.health)
                    s_dd2 = State(self.pos , self.mat,self.arr,self.state,self.health)
                    s_dr1 = State(self.pos , min(2,self.mat + 1),self.arr,"R",self.health)
                    s_dr2 = State(self.pos , self.mat ,self.arr,"R",self.health)
                    return [
                        (p_stayd*p_succ_getmat , s_dd1),
                        (p_stayd*(1 - p_succ_getmat) , s_dd2),
                        (p_dtor*p_succ_getmat , s_dr1),
                        (p_dtor*(1 - p_succ_getmat) , s_dr2)
                    ]

                if self.state == "R":
                    s_ra1 = State(self.pos , min(2,self.mat + 1),self.arr,self.state,self.health)
                    s_ra2 = State(self.pos , self.mat,self.arr,self.state,self.health)
                    s_rn1 = State(self.pos , min(2,self.mat + 1),self.arr,"D",self.health)
                    s_rn2 = State(self.pos , self.mat ,self.arr,"D",self.health)
                    return [
                        (p_attack*p_succ_getmat , s_ra1),
                        (p_noAtk*(1 - p_succ_getmat) , s_ra2),
                        (p_attack*p_succ_getmat , s_rn1),
                        (p_noAtk*(1 - p_succ_getmat) , s_rn2)
                    ]
    
    def output_state(self):
        return [self.pos,self.mat,self.arr,self.state,self.health]

# calculation final dimension
final_dimension = 0
for x1 in POS_DIR:
    for x2 in range(MAT_NUM):
        for x3 in range(ARR_NUM):
            for x4 in STATEMM_ARR:
                for x5 in HEALTH_ARR:
                    final_dimension = final_dimension + len(State(x1,x2,x3,x4,x5).Actions())

r = np.zeros((1, final_dimension))
a = np.zeros((TOTAL_NUM_STATE , final_dimension))
alpha = np.zeros((TOTAL_NUM_STATE , 1))

# for alpha
index_alpha = 120 * POS_DIR.index(current_pos) + 40 * current_mat + 10 * current_arrow + 5 * STATEMM_ARR.index(current_state) + HEALTH_ARR.index(current_health)
alpha[index_alpha][0] = 1.0

# calculating r
indexr = 0
for x1 in POS_DIR:
    for x2 in range(MAT_NUM):
        for x3 in range(ARR_NUM):
            for x4 in STATEMM_ARR:
                for x5 in HEALTH_ARR:
                    state = State(x1,x2,x3,x4,x5)
                    actions = state.Actions()

                    for action in actions:
                        if action == 0:
                            r[0][indexr] = 0
                            indexr = indexr + 1
                        
                        elif action >= 1 and action <= 4:
                            r[0][indexr] = step_cost
                            indexr = indexr + 1
                            if (x1 == "C" or x1 == "E") and x4 == "R":
                                r[0][indexr - 1] = r[0][indexr - 1] + p_attack * mm_hit_reward

                        elif action == 5:
                            r[0][indexr] = stay_cost
                            indexr = indexr + 1
                            if (x1 == "C" or x1 == "E") and x4 == "R":
                                r[0][indexr - 1] = r[0][indexr - 1] + p_attack * mm_hit_reward

                        elif action == 6 or action == 7:
                            r[0][indexr] = step_cost
                            indexr = indexr + 1
                            if (x1 == "C" or x1 == "E") and x4 == "R":
                                r[0][indexr - 1] = r[0][indexr - 1] + p_attack * mm_hit_reward

                        elif action == 8 or action == 9:
                            r[0][indexr] = step_cost
                            indexr = indexr + 1

indexa = 0
for x1 in POS_DIR:
    for x2 in range(MAT_NUM):
        for x3 in range(ARR_NUM):
            for x4 in STATEMM_ARR:
                for x5 in HEALTH_ARR:
                    state = State(x1,x2,x3,x4,x5)
                    actions = state.Actions()
                    index_state = POS_DIR.index(x1) * 120 + x2 * 40 + x3 * 10 + STATEMM_ARR.index(x4)*5 + HEALTH_ARR.index(x5) * 1
                    for action in actions:
                        new_states = state.simulate(action)

                        if len(new_states) == 0:
                            a[index_state][indexa] = 1

                        for new_state in new_states:
                            index_new_state = POS_DIR.index(new_state[1].pos) * 120 + new_state[1].mat * 40 + new_state[1].arr * 10 + STATEMM_ARR.index(new_state[1].state) * 5 + HEALTH_ARR.index(new_state[1].health) * 1
                            a[index_new_state][indexa] -= new_state[0]
                            a[index_state][indexa] = a[index_state][indexa] + new_state[0]

                        indexa = indexa + 1

# cvpxy calculations
print("Dim " + str(final_dimension))
print("TotalStates " + str(TOTAL_NUM_STATE))
# print("a")
# for i in range(600):
    # print(a[i][1])
# print("r")
# print(np.shape(r))
# print("Alpha")
# print(alpha)

x = cp.Variable((final_dimension, 1), 'x')
        
constraints = [
    cp.matmul(a, x) == alpha,
    x >= 0
]

objective = cp.Maximize(cp.matmul(r, x))
problem = cp.Problem(objective, constraints)

solution = problem.solve()
arr_x = list(x.value)
l = [ float(val) for val in arr_x]


# lets define policy
policy = []
index_policy = 0
for x1 in POS_DIR:
    for x2 in range(MAT_NUM):
        for x3 in range(ARR_NUM):
            for x4 in STATEMM_ARR:
                for x5 in HEALTH_ARR:
                    state = State(x1,x2,x3,x4,x5)
                    actions = state.Actions()
                    index_action = np.argmax(l[index_policy : index_policy+len(actions)])
                    index_policy += len(actions)
                    best_action = actions[index_action]
                    policy.append(state.output_state())
                    key_list = list(ACTIONS.keys())
                    val_list = list(ACTIONS.values())
                    index_val = val_list.index(best_action)
                    key_list_value = key_list[index_val]
                    policy.append(key_list_value)


solDict = {}
solDict["a"] = a.tolist()
r = [float(val) for val in np.transpose(r)]
solDict["r"] = r
solDict["alpha"] = alpha.tolist()
solDict["x"] = l
solDict["policy"] = policy
solDict["objective"] = float(solution)
os.makedirs('outputs', exist_ok=True)
path = "outputs/part_3_output.json"
json_object = json.dumps(solDict, indent=4)
with open(path, 'w+') as f:
  f.write(json_object)
print(solution)
 
