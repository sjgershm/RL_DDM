function data = load_bandit_data
    
    % Load data from two-armed bandit task.
    %
    % USAGE: data = load_bandit_data
    %
    % OUTPUTS:
    %   data - [S x 1] structure, where S is the number of subjects, with the following fields:
    %           .c - [N x 1] choices
    %           .r - [N x 1] rewards
    %           .s - [N x 1] states
    %           .conf - [N x 1] confidence judgment
    %           .prob - [N x 2] reward probabilities
    %           .rt - [N x 1] response times
    %           .C - number of choice options
    %           .N - number of trials
    
    D = csvread('data_bandit.csv',1);
    
    subs = unique(D(:,1));      % subjects
    
    for i = 1:length(subs)
        ix = D(:,1)==subs(i);
        data(i).prob = D(ix,2:3);
        data(i).s = D(ix,4);
        data(i).c = D(ix,5);
        data(i).r = D(ix,6);
        data(i).rt = D(ix,7);
        data(i).conf = D(ix,8);
        data(i).C = 2;
        data(i).N = length(data(i).c);
    end