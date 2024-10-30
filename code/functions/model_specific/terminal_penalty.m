function penalty_out = terminal_penalty(xEnd, yEnd)
global gama sigma x_targ Ks
penalty_out = gama*(1-exp(- (xEnd-x_targ(1)/Ks).^2/sigma  - (yEnd-x_targ(2)).^2/sigma ));
end