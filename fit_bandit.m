function results = fit_bandit(data)
    
    % Fit RL-DDM model to data from a two-armed bandit task.
    %
    % USAGE: results = fit_bandit(data)
    %
    % INPUTS:
    %   data - [S x 1] data structure, where S is the number of subjects; see likfun_bandit for more details
    %
    % OUTPUTS:
    %   results - see mfit_optimize for more details
    %
    % Sam Gershman, Jun 2016
    
    % create parameter structure
    
    % drift rate differential action value weight
    param(1).name = 'b';
    param(1).logpdf = @(x) 0;  % uniorm prior
    param(1).lb = -20; % lower bound
    param(1).ub = 20;   % upper bound
    
    % learning rate
    param(2).name = 'lr';
    param(2).hp = [1.2 1.2];    % hyperparameters of beta prior
    param(2).logpdf = @(x) sum(log(betapdf(x,param(2).hp(1),param(2).hp(2))));
    param(2).lb = 0;
    param(2).ub = 1;
    
    % decision threshold
    param(3).name = 'a';
    param(3).logpdf = @(x) 0;
    param(3).lb = 1e-3;
    param(3).ub = 20;
    
    % non-decision time
    param(4).name = 'T';
    param(4).logpdf = @(x) 0;
    param(4).lb = 0;
    param(4).ub = 1;
    
    % fit model
    f = @(x,data) likfun_bandit(x,data);    % log-likelihood function
    results = mfit_optimize(f,param,data);