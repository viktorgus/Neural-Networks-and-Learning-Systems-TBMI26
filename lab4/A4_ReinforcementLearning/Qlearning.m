%% Initialization
%  Initialize the world, Q-table, and hyperparameters

episodes = 1000;
learningrate = 0.1;
discountfactor = 0.999;
k = 1; %world index

world = gwinit(k);
Q = rand(world.xsize(),world.ysize(),4); %init q table

totIt = 0;
movAvg = 0;
%% Training loop
%  Train the agent using the Q-learning algorithm.

for episode = 1:episodes
    world = gwinit(k);
    
    epsilon = getepsilon(episode,episodes);
   
    state = gwstate();
    P = getpolicy(Q);
    it = 0;
    while state.isterminal == 0
       P = getpolicy(Q);
       
       it = it + 1;
       action = chooseaction(Q,state.pos(2),state.pos(1),[1,2,3,4],[1,1,1,1],epsilon); 
       old_x = state.pos(2); 
       old_y = state.pos(1); 
       
       
       newstate = gwaction(action);
       
       while newstate.isvalid == 0
            Q(old_x, old_y, action) = -inf;
            action = chooseaction(Q,old_x,old_y,[1,2,3,4],[1,1,1,1],1); 
            newstate = gwaction(action);
            
       end
        
       
       new_x = newstate.pos(2); 
       new_y = newstate.pos(1); 
       
       reward = newstate.feedback;
       
       V = getvalue(Q);
       Q(old_x, old_y, action) = (1-learningrate)*Q(old_x, old_y, action)+learningrate*(reward+discountfactor*V(new_x, new_y));
        
       
      % gwdraw(episode, P);
       state = newstate;
        
    end
    totIt = totIt + it;
    movAvg = totIt/episode;
    
    if mod(episode,100)==0
    disp(movAvg);
    end

end


%% Test loop
%  Test the agent (subjectively) by letting it use the optimal policy
%  to traverse the gridworld. Do not update the Q-table when testing.
%  Also, you should not explore when testing, i.e. epsilon=0; always pick
%  the optimal action.


 P = getpolicy(Q);
 gwdraw(episode, P);

 
 epsilon = 0; 
 test_episodes = 10;
 iterations = zeros(test_episodes,1);
 maxTries = 200;
disp('Testing');

for i=1:test_episodes
    gwinit(k);
    state = gwstate();
while state.isterminal == 0 && iterations(i)<maxTries 
        
        iterations(i) = iterations(i)+1;

       [~, opt_action] = chooseaction(Q,state.pos(2),state.pos(1),[1,2,3,4],[1,1,1,1],epsilon); 
       old_x = state.pos(2); 
       old_y = state.pos(1); 
       
       
       state = gwaction(opt_action);
       gwdraw(i,P);
       
       
    
       
end
 
end

avg_iterations = sum(iterations)/test_episodes