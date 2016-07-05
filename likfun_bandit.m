function [lik, latents] = likfun_bandit(x,data)
    
    % Likelihood function for two-armed bandit task.
    
    % USAGE: [lik, latents] = likfun_bandit(x,data)
    %
    % INPUTS:
    %   x - parameters:
    %       x(1) - drift rate differential action value weight (b)
    %       x(2) - learning rate for state-action values (alpha)
    %       x(3) - decision threshold (a)
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
    %           .P - [N x 1] action probability
    %           .v - [N x 1] drift rate
    %
    % Sam Gershman, Nov 2015
    
    % set parameters
    b = x(1);           % drift rate differential action value weight
    lr = x(2);          % learning rate
    a = x(3);           % decision threshold
    
    % initialization
    lik = 0; C = data.C;
    S = length(unique(data.s)); % number of states
    Q = zeros(S,C);    % initial state-action values
    
    for n = 1:data.N
        
        % data for current trial
        c = data.c(n);              % choice
        r = data.r(n);              % reward
        s = data.s(n);              % state
        
        % drift rate
        v = b*(Q(s,2)-Q(s,1));
        
        % accumulate log-likelihod
        if data.c(n) == 2
            P = wfpt(data.rt(n),-v,a);  % Wiener first passage time distribution
        else
            P = wfpt(data.rt(n),v,a);
        end
        if isnan(P) || P==0; P = realmin; end % avoid NaNs and zeros in the logarithm
        lik = lik + log(P);
        
        % update values
        Q(s,c) = Q(s,c) + lr*(r - Q(s,c));
        
        % store latent variables
        if nargout > 1
            latents.v(n,1) = v;
            latents.P(n,1) = P;
        end
        
    end