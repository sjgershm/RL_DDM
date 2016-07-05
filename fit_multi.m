function results = fit_multi(data)
    
    % Fit multi-alternative DDM model to data from a value-based choice task.
    %
    % USAGE: results = fit_multi(data)
    %
    % INPUTS:
    %   data - [S x 1] data structure, where S is the number of subjects; see likfun_multi for more details
    %
    % OUTPUTS:
    %   results - see mfit_optimize for more details
    %
    % Sam Gershman, Jul 2016
    
    % create parameter structure
    
    % drift rate differential action value weight
    param(1).name = 'b';
    param(1).logpdf = @(x) 0;  % uniorm prior
    param(1).lb = -20; % lower bound
    param(1).ub = 20;   % upper bound
    
    % decision threshold
    param(2).name = 'a';
    param(2).logpdf = @(x) 0;
    param(2).lb = 1e-3;
    param(2).ub = 20;
    
    % fit model
    f = @(x,data) likfun_multi(x,data);    % log-likelihood function
    results = mfit_optimize(f,param,data);
    
    % get latent variables
    for i = 1:length(data)
        [~,results.latents(i)] = likfun_bandit(results.x(i,:),data(i));
    end