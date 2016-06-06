function data = load_gonogo_data
    
    % Load data from Go/NoGo task.
    %
    % USAGE: data = load_gonogo_data
    %
    % OUTPUTS:
    %   data - [S x 1] structure, where S is the number of subjects, with the following fields:
    %           .c - [N x 1] choices
    %           .r - [N x 1] rewards
    %           .s - [N x 1] states
    %           .go - [N x 1] go trial indicator (1=Go, 0=NoGo)
    %           .rt - [N x 1] response times
    %           .C - number of choice options
    %           .N - number of trials
    
    D = csvread('data_gonogo.csv',1);
    
    subs = unique(D(:,1));      % subjects
    
    for i = 1:length(subs)
        ix = D(:,1)==subs(i);
        data(i).go = D(ix,2);
        data(i).s = D(ix,3);
        data(i).c = D(ix,4);
        data(i).r = D(ix,5);
        data(i).rt = D(ix,6);
        data(i).C = 2;
        data(i).N = length(data(i).c);
    end