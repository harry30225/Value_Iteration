# Part 3 (Liner Programming)

### Team 73
* Harshit Sharma (2019101083)
* Aaditya Sharma (2019113009)

## How A Matrix is formed
The dimensions of A matrix are TOTAL_NUM_STATES * FULL_DIMENSION  
where TOTAL_NUM_STATES is the number of valid states formed by   
Data of states
POS_NUM = 5       # W,N,S,E,C  
MAT_NUM = 3       # 0,1,2  
ARR_NUM = 4       # 0,1,2,3  
STATE_NUM = 2     # D,R  
HEALTH_NUM = 5    # 0,25,50,75,100  
TOTAL_NUM_STATE = POS_NUM * MAT_NUM * ARR_NUM * STATE_NUM * HEALTH_NUM  

FULL_DIMENSION are the number of total actions in going from one state to another in all the valid states  

Now iterating for each state, i will first find all the valid actions of state. Then according to each action , i determine the next state  
and the probablilty associated with the state. Now comparing it with a flow of water from pipes problem , i am assigning values to each cell  

Initially the array is initialized with zeroes. Now for a action, the probability assigned to the action is incremented into the initial  
state and decremented from the next state reached by performing that action. The actions are according to the dormat or ready state  
of MM and different actions and also the validity of action is checked. If the ACTION is a NOOP, then 1 is added to the initial state as  
there is no next state.In this way the A matrix was formed. This matrix basically stores thetransition probabilities, a kind of.  
Like how much change will occur to a state by a following action that is captured in this matrix.  

## Explain the procedure of finding the policy and explain the results?
To the find the policy, we need to look at actions associated with each state and find the best one. Now as we solved an LP, which is  
solving max(rx) st Ax = alpha. x vector which basically maximizes the avg reward that we will get till the end of the game.  
 In this X we have the probabilities by which each action is selected in each state.Now i would find the action that has maximum  
contribution in the state, that would be selected. Like this our policy would be created.  
if The current policy that we have got by assuming the start state to be [“C”,2,3,”R”,100] says to move to NORTH by  
choosing UP action and then opting to craft if i can and i have number of arrows such that 25*num_arrows < MM_health then this is preferred.  

## Can there be multiple policies?
Yes , there could be a number of policies, it depends on the number of factors. If we change the current state, the policy would be affected  
As it would lead to change in alpha vector. Now if we alter our rewards, step_cost etc. the also we would encounter a different policy   
as changing these changes r vector. Now, if we alter probabilities of different actions, this would also lead to a different policy due to   
changes A vector, thus changing any vector out of r,aplha and A would change the policy as it alters the LP performed.