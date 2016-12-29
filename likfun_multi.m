function [lik, latents] = likfun_multi(x,data)
    
    % Likelihood function for value-based decision-making tasks with
    % multiple alternatives.
    
    % USAGE: [lik, latents] = likfun_multi(x,data)
    %
    % INPUTS:
    %   x - parameters:
    %       x(1) - drift rate value weight (b)
    %       x(2) - decision threshold (a)
    %       x(3) - non-decision time (T)
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
    %           .v - [N x C] drift rates
    %
    % Sam Gershman, Nov 2015
    
    % set parameters
    b = x(1);           % drift rate action value weight
    a = x(2);           % decision threshold
    T = x(3);           % non-decision time
    
    % initialization
    lik = 0; C = data.C;
    data.rt = max(eps,data.rt - T);
    
    for n = 1:data.N
        
        % data for current trial
        c = data.c(n);              % choice
        v = b*data.V(n,:);          % drift rate
        rt = data.rt(n);            % response time
        
        % accumulate log-likelihood
        logP = log(wfpt(rt,-v(c),a));
        for k = 1:C
            if k~=c
                logP = logP + log(max(realmin,1-integral(@(t) wfpt(t,-v(k),a),0,rt)));
            end
        end
        if isnan(logP) || isinf(logP) || ~isreal(logP); logP = log(realmin); end
        lik = lik + logP;
        
        % store latent variables
        if nargout > 1
            latents.v(n,:) = v;
        end
        
    end