function [lik, latents] = likfun_binary(x,data)
    
    % Likelihood function for binary value-based decision-making tasks.
    
    % USAGE: [lik, latents] = likfun_binary(x,data)
    %
    % INPUTS:
    %   x - parameters:
    %       x(1) - drift rate value weight (b)
    %       x(2) - decision threshold (a)
    %       x(3) - non-decision time (T)
    %   data - structure with the following fields
    %           .c - [N x 1] choices (c=1: higher value option was chosen, c=0: lower value option was chosen)
    %           .V - [N x 1] difference in value between two options (higher - lower)
    %           .rt - [N x 1] response times
    %           .N - number of trials
    %
    % OUTPUTS:
    %   lik - log-likelihood
    %   latents - structure with the following fields:
    %           .v - [N x 1] drift rates
    %           .P - [N x 1] probability of chosen option
    %           .RT_mean - [N x 1] mean response time for chosen option
    %
    % Sam Gershman, Dec 2016
    
    % set parameters
    b = x(1);           % drift rate action value weight
    a = x(2);           % decision threshold
    T = x(3);           % non-decision time
    if length(x)>3; b0 = x(4); else b0 = 0; end
    
    % initialization
    lik = 0;
    data.rt = max(eps,data.rt - T);
    
    for n = 1:data.N
        
        % accumulate log-likelihood
        v = b0 + b*data.V(n);     % drift rate
        if data.c(n) == 1; v = -v; end
        P = wfpt(data.rt(n),-v,a);  % Wiener first passage time distribution
        if isnan(P) || P==0; P = realmin; end % avoid NaNs and zeros in the logarithm
        lik = lik + log(P);
        
        % store latent variables
        if nargout > 1
            latents.v(n,:) = v;
            latents.P(n,1) = 1/(1+exp(-a*v));
            latents.RT_mean(n,1) = (0.5*a/v)*tanh(0.5*a*v)+T;
        end
        
    end