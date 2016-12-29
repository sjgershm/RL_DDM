function [lik, latents] = likfun_bandit(x,data)
    
    % Likelihood function for two-armed bandit task.
    
    % USAGE: [lik, latents] = likfun_bandit(x,data)
    %
    % INPUTS:
    %   x - parameters:
    %       x(1) - drift rate differential action value weight (b)
    %       x(2) - learning rate for state-action values (alpha)
    %       x(3) - decision threshold (a)
    %       x(4) - non-decision time (T)
    %   data - structure with the following fields
    %           .c - [N x 1] choices
    %           .r - [N x 1] rewards
    %           .s - [N x 1] states
    %           .rt - [N x 1] response times
    %           .C - number of choice options
    %           .N - number of trials
    %
    % OUTPUTS:
    %   lik - log-likelihood
    %   latents - structure with the following fields:
    %           .v - [N x 1] drift rate
    %           .P - [N x 1] probability of chosen option
    %           .RT_mean - [N x 1] mean response time for chosen option
    %
    % Sam Gershman, Aug 2016
    
    % set parameters
    b = x(1);           % drift rate differential action value weight
    lr = x(2);          % learning rate
    a = x(3);           % decision threshold
    T = x(4);           % non-decision time
    
    % initialization
    lik = 0; C = data.C;
    S = length(unique(data.s)); % number of states
    Q = zeros(S,C);    % initial state-action values
    data.rt = max(eps,data.rt - T);
    
    for n = 1:data.N
        
        % data for current trial
        c = data.c(n);              % choice
        r = data.r(n);              % reward
        s = data.s(n);              % state
        
        % drift rate
        v = b*(Q(s,2)-Q(s,1));
        
        % accumulate log-likelihod
        if data.c(n) == 1; v = -v; end
        P = wfpt(data.rt(n),-v,a);  % Wiener first passage time distribution
        if isnan(P) || P==0; P = realmin; end % avoid NaNs and zeros in the logarithm
        lik = lik + log(P);
        
        % update values
        Q(s,c) = Q(s,c) + lr*(r - Q(s,c));
        
        % store latent variables
        if nargout > 1
            latents.v(n,1) = v;
            latents.P(n,1) = 1/(1+exp(-a*v));
            latents.RT_mean(n,1) = (0.5*a/v)*tanh(0.5*a*v)+T;
        end
        
    end