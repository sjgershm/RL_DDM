function [lik, latents] = likfun_logreg(x,data)
    
    % Likelihood function for logistic regression.
    
    % USAGE: [lik, latents] = likfun_binary(x,data)
    %
    % INPUTS:
    %   x - regression coefficients
    %   data - structure with the following fields
    %           .c - [N x 1] choices (c=1: higher value option was chosen, c=0: lower value option was chosen)
    %           .V - [N x 1] difference in value between two options (higher - lower)
    %           .X - [N x K] nuisance regressors
    %           .N - number of trials
    %
    % OUTPUTS:
    %   lik - log-likelihood
    %   latents - structure with the following fields:
    %           .P - [N x 1] probability of chosen option
    %
    % Sam Gershman, Dec 2016
    
    v = [ones(data.N,1) data.V data.X]*x';
    p = 1./(1+exp(-v));
    lik = sum(data.c.*log(p) + (1-data.c).*log(1-p));
    
    % store latent variables
    if nargout > 1
        latents.P = p;
    end