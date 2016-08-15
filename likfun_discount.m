function [lik, latents] = likfun_discount(x,data)
    
    % Likelihood function for hyperbolic discounting.
    
    % USAGE: [lik, latents] = likfun_discount(x,data)
    %
    % INPUTS:
    %   x - parameters:
    %       x(1) - drift rate differential action value weight (b)
    %       x(2) - discount parameter (k)
    %       x(3) - decision threshold (a)
    %   data - structure with the following fields
    %           .c - [N x 1] choices
    %           .r - [N x 2] reward for each option
    %           .d - [N x 2] delay for each option
    %           .rt - [N x 1] response times (seconds)
    %           .C - number of choice options
    %           .N - number of trials
    %
    % OUTPUTS:
    %   lik - log-likelihood
    %   latents - structure with the following fields:
    %           .lik - [N x 1] likelihood
    %           .P - [N x 1] probability of choosing option 1
    %           .RT_mean - [N x 1] mean response time
    %           .v - [N x 1] drift rate
    %
    % Sam Gershman, Nov 2015
    
    % set parameters
    b = x(1);           % drift rate differential action value weight
    k = x(2);           % discount parameter
    a = x(3);           % decision threshold
    
    lik = 0;
    for n = 1:data.N
        
        % data for current trial
        r = data.r(n,:);            % rewards
        d = data.d(n,:);            % delays
        
        % drift rate
        V = r./(1+k*d);
        v = b*(diff(V));
        
        % accumulate log-likelihod
        if data.c(n) == 2
            P = wfpt(data.rt(n),-v,a);  % Wiener first passage time distribution
        else
            P = wfpt(data.rt(n),v,a);
        end
        if isnan(P) || P==0; P = realmin; end % avoid NaNs and zeros in the logarithm
        lik = lik + log(P);
        
        % store latent variables
        if nargout > 1
            latents.v(n,1) = v;
            latents.lik(n,1) = P;
            latents.P(n,1) = 1/(1+exp(-a*v));
            latents.RT_mean(n,1) = (0.5*a/v)*tanh(0.5*a*v);
        end
        
    end