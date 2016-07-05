function [lik, latents] = likfun_multi(x,data)
    
    % Likelihood function for value-based decision-making tasks with
    % multiple alternatives.
    
    % USAGE: [lik, latents] = likfun_multi(x,data)
    %
    % INPUTS:
    %   x - parameters:
    %       x(1) - drift rate value weight (b)
    %       x(2) - decision threshold (a)
    %   data - structure with the following fields
    %           .c - [N x 1] choices
    %           .V - [N x C] values
    %           .rt - [N x 1] response times
    %           .C - number of choice options
    %           .N - number of trials
    %
    % OUTPUTS:
    %   lik - log-likelihood
    %   latents - structure with the following fields:
    %           .logP - [N x 1] action log probability
    %           .v - [N x C] drift rates
    %
    % Sam Gershman, Nov 2015
    
    % set parameters
    b = x(1);           % drift rate differential action value weight
    a = x(2);           % decision threshold
    
    % initialization
    lik = 0; C = data.C;
    
    for n = 1:data.N
        
        % data for current trial
        c = data.c(n);              % choice
        v = b*data.V(n,:);          % drift rate
        rt = data.rt(n);            % response time
        
        % accumulate log-likelihood
        logP = wfpt(t,-v(c),a);
        for k = 1:C
            if k~=c
                logP = logP + log(1-integral(@(t) wfpt(t,-v(k),a),0,rt));
            end
        end
        if isnan(logP); logP = log(realmin); end
        lik = lik + logP;
        
        % store latent variables
        if nargout > 1
            latents.v(n,:) = v;
            latents.logP(n,1) = logP;
        end
        
    end