function results = fit_gonogo(data)
    
    % Fit RL-DDM model to data from a Go/NoGo task.
    %
    % USAGE: results = fit_gonogo(data)
    %
    % INPUTS:
    %   data - [S x 1] data structure, where S is the number of subjects; see likfun_bandit for more details
    %
    % OUTPUTS:
    %   results - see mfit_optimize for more details
    %
    % Sam Gershman, Jun 2016
    
    % create parameter structure
    
    % drift rate go bias weight
    param(1).name = 'b1';
    param(1).logpdf = @(x) 0;  % uniorm prior
    param(1).lb = -20; % lower bound
    param(1).ub = 20;   % upper bound
    
    % drift rate differential action value weight
    param(2) = param(1);
    param(2).name = 'b2';
    
    % drift rate Pavlovian bias weight
    param(3) = param(1);
    param(3).name = 'b3';
    
    % learning rate
    param(4).name = 'lr_pos';
    param(4).hp = [1.2 1.2];    % hyperparameters of beta prior
    param(4).logpdf = @(x) sum(log(betapdf(x,param(4).hp(1),param(4).hp(2))));
    param(4).lb = 0;
    param(4).ub = 1;
    
    % decision threshold
    param(5).name = 'a';
    param(5).logpdf = @(x) 0;
    param(5).lb = 1e-3;
    param(5).ub = 20;
    
    % non-decision time
    param(6).name = 'T';
    param(6).logpdf = @(x) 0;
    param(6).lb = 0;
    param(6).ub = 1;
    
    % fit model
    f = @(x,data) likfun_gonogo(x,data);    % log-likelihood function
    results = mfit_optimize(f,param,data);