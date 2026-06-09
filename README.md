# anna_phd
to dithering_new_notch:

One thing I couldn't do here: actually execute it, since this needs MATLAB (the arguments block, islocalmax, fminbnd). If islocalmax ever returns the same peak twice on a very flat merged profile, the bracket collapses and fminbnd returns that point — if you see that in practice, add a xPk(end)-xPk(1) > someMinSep guard before the fminbnd call.
