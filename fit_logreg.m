function results = fit_logreg(data)
    
    % Fit two-alternative logistic regression model to data from a value-based choice task.
    %
    % USAGE: results = fit_logreg(data)
    %
    % INPUTS:
    %   data - [S x 1] data structure, where S is the number of subjects.
    %   Each element has the following fields:
    %           .c - [N x 1] choices (c=1: higher value option was chosen, c=0: lower value option was chosen)
    %           .V - [N x 1] difference in value between two options (higher - lower)
    %           .X - [N x K] nuisance regressors
    %           .N - number of trials
    %
    % OUTPUTS:
    %   results - see mfit_optimize for more details
    %
    % Sam Gershman, Dec 2016
    
    % intercept
    param(1).name = 'b0';
    param(1).logpdf = @(x) 0;  % uniorm prior
    param(1).lb = -20; % lower bound
    param(1).ub = 20;   % upper bound
    
    % value difference coefficient
    param(2).name = 'b1';
    param(2).logpdf = @(x) 0;  % uniorm prior
    param(2).lb = -20; % lower bound
    param(2).ub = 20;   % upper bound
    
    % nuisance regressor coefficients
    if isfield(data(1),'X')
        for k = 1:size(data(1).X,2)
            param(k+2).name = ['b',num2str(k+1)];
            param(k+2).logpdf = @(x) 0;
            param(k+2).lb = -20;
            param(k+2).ub = 20;
        end
    end
    
    % fit model
    f = @(x,data) likfun_logreg(x,data);    % log-likelihood function
    results = mfit_optimize(f,param,data);