classdef geluLayer < nnet.layer.Layer
    
    % Copyright 2021 The MathWorks, Inc.
    
    methods
        function layer = geluLayer(args) 
            arguments
                args.Name = "";
            end
    
            % Set layer name.
            layer.Name = args.Name;
        end
        
        function Z = predict(~,x)
            Z = 0.5*x.*(1 + tanh(sqrt(2/pi)*(x + 0.044715*(x.^3))));
        end
    end
end
